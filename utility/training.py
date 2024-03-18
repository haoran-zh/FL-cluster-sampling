import torch
from torch.utils.data import Subset, DataLoader
from utility.load_model import load_model
from utility.config import optimizer_config
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import utility.optimal_sampling as optimal_sampling
import numpy as np

def training(tasks_data_info, tasks_data_idx, global_models, chosen_clients, task_type, clients_task, local_epochs, batch_size, classes_size, type_iid, device, args):
    all_tasks_weights_list = []
    all_tasks_local_training_acc = []
    all_tasks_local_training_loss = []
    all_weights_diff = []
    for tasks_index in range(len(task_type)): # all tasks and all clients need to be trained
        tasks_weights_list = []
        tasks_local_training_acc = []
        tasks_local_training_loss = []
        weights_diff = []
        for client_index in range(args.num_clients):
            local_model = load_model(name_data=task_type[int(tasks_index)], num_classes=classes_size[tasks_index][5],
                                     args=args).to(device)
            global_model_state_dict = global_models[int(tasks_index)].state_dict()
            local_model_state_dict = local_model.state_dict()
            for key in local_model_state_dict.keys():
                local_model_state_dict[key] = global_model_state_dict[key]
            local_model.load_state_dict(local_model_state_dict)
            previous_local_state_dict = local_model_state_dict.copy()
            learning_rate = args.lr
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
            local_criterion = nn.CrossEntropyLoss()
            # Get client's data
            if type_iid[tasks_index] == 'iid':
                client_data = Subset(tasks_data_info[tasks_index][0], tasks_data_idx[tasks_index][client_index])  # or iid_partition depending on your choice
            elif type_iid[tasks_index] == 'noniid':
                client_data = Subset(tasks_data_info[tasks_index][0], tasks_data_idx[tasks_index][0][client_index])  # or iid_partition depending on your choice
                client_label = tasks_data_idx[tasks_index][1][client_index]
            client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
            local_model.train()
            local_train_accuracy = 0
            for _ in range(local_epochs[tasks_index]):
                correct = 0
                running_loss = 0
                total = 0
                for images, labels in client_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    local_optimizer.zero_grad()
                    outputs = local_model(images)
                    if type_iid[tasks_index] == 'noniid':
                        label_mask = torch.zeros(classes_size[tasks_index][5], device=outputs.device)
                        label_mask[client_label] = 1
                        outputs = outputs.masked_fill(label_mask == 0, 0)
                    loss = local_criterion(outputs, labels)
                    if client_index in chosen_clients:
                        if tasks_index == clients_task[chosen_clients.index(client_index)]:
                            loss.backward()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    running_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    local_optimizer.step()
                local_train_accuracy = correct / total
                local_train_loss = running_loss / total

            tasks_local_training_acc.append(local_train_accuracy)
            tasks_local_training_loss.append(local_train_loss)
            # Append local model weights to list
            norm, lr_gradients = optimal_sampling.get_gradient_norm(previous_local_state_dict, local_model.state_dict(), args)
            # new lr = original_lr * sum(f^alpha-1(w_t))
            weights_diff.append(norm)

            if args.cpumodel is True:
                local_model.to('cpu')
            tasks_weights_list.append(lr_gradients.copy())


            # take a step once every global epoch
            # scheduler.step()
        all_tasks_weights_list.append(tasks_weights_list)
        all_tasks_local_training_acc.append(tasks_local_training_acc)
        all_tasks_local_training_loss.append(tasks_local_training_loss)
        all_weights_diff.append(weights_diff)
    # remove the inactive clients
    tasks_weights_list = []
    tasks_local_training_acc = []
    tasks_local_training_loss = []
    weights_diff = []

    for i in range(len(chosen_clients)):
        active_client_index = chosen_clients[i]
        task_index = clients_task[i]
        tasks_weights_list.append(all_tasks_weights_list[task_index][active_client_index])
        tasks_local_training_acc.append(all_tasks_local_training_acc[task_index][active_client_index])
        tasks_local_training_loss.append(all_tasks_local_training_loss[task_index][active_client_index])
        weights_diff.append(all_weights_diff[task_index][active_client_index])


    return tasks_weights_list, tasks_local_training_acc, tasks_local_training_loss, weights_diff, all_tasks_local_training_loss


def training_all(tasks_data_info, tasks_data_idx, global_models, chosen_clients, task_type, clients_task, local_epochs,
             batch_size, classes_size, type_iid, device, args):
    all_tasks_weights_list = []
    all_tasks_local_training_acc = []
    all_tasks_local_training_loss = []
    all_weights_diff = []
    # we want to get a list of weights, weights_list[task_index][client_index].
    # client_index is in the order of chosen_clients

    for tasks_index in range(len(task_type)): # all tasks and all clients need to be trained
        tasks_weights_list = []
        tasks_local_training_acc = []
        tasks_local_training_loss = []
        weights_diff = []
        for client_index in range(args.num_clients):
            local_model = load_model(name_data=task_type[int(tasks_index)], num_classes=classes_size[tasks_index][5],
                                     args=args).to(device)

            global_model_state_dict = global_models[int(tasks_index)].state_dict()
            local_model_state_dict = local_model.state_dict()

            for key in local_model_state_dict.keys():
                local_model_state_dict[key] = global_model_state_dict[key]
            local_model.load_state_dict(local_model_state_dict)

            previous_local_state_dict = local_model_state_dict.copy()

            # Create a local optimizer
            learning_rate, momentum, weight_decay, lr_step_size, gamma, milestones = optimizer_config(task_type[tasks_index])
            learning_rate = args.lr
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
            # learning rate scheduler
            # scheduler = lr_scheduler.StepLR(local_optimizer, step_size=lr_step_size, gamma=gamma)
            # scheduler = lr_scheduler.MultiStepLR(local_optimizer, milestones=milestones, gamma=gamma)
            local_criterion = nn.CrossEntropyLoss()

            # Get client's data
            if type_iid[tasks_index] == 'iid':
                client_data = Subset(tasks_data_info[tasks_index][0], tasks_data_idx[tasks_index][client_index])  # or iid_partition depending on your choice
            elif type_iid[tasks_index] == 'noniid':
                client_data = Subset(tasks_data_info[tasks_index][0], tasks_data_idx[tasks_index][0][client_index])  # or iid_partition depending on your choice
                client_label = tasks_data_idx[tasks_index][1][client_index]
            client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
            local_model.train()
            local_train_accuracy = 0

            for _ in range(local_epochs[tasks_index]):
                correct = 0
                running_loss = 0
                total = 0
                for images, labels in client_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    local_optimizer.zero_grad()
                    outputs = local_model(images)
                    if type_iid[tasks_index] == 'noniid':
                        label_mask = torch.zeros(classes_size[tasks_index][5], device=outputs.device)
                        label_mask[client_label] = 1
                        outputs = outputs.masked_fill(label_mask == 0, 0)
                    # print(outputs.shape)
                    # print(labels.shape)
                    loss = local_criterion(outputs, labels)
                    loss.backward()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    running_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    local_optimizer.step()
                local_train_accuracy = correct / total
                local_train_loss = running_loss / total

            tasks_local_training_acc.append(local_train_accuracy)
            tasks_local_training_loss.append(local_train_loss)
            # Append local model weights to list
            norm, lr_gradients = optimal_sampling.get_gradient_norm(previous_local_state_dict, local_model.state_dict(), args)
            # new lr = original_lr * sum(f^alpha-1(w_t))
            weights_diff.append(norm)

            if args.cpumodel is True:
                local_model.to('cpu')
            tasks_weights_list.append(lr_gradients.copy())


            # take a step once every global epoch
            # scheduler.step()
        all_tasks_weights_list.append(tasks_weights_list)
        all_tasks_local_training_acc.append(tasks_local_training_acc)
        all_tasks_local_training_loss.append(tasks_local_training_loss)
        all_weights_diff.append(weights_diff)
        # all_xxx list is in shape: [task_index][client_index]
        # client_index is in the order of chosen_clients[0], chosen_clients[1], chosen_clients[2]...
    return all_tasks_weights_list, all_tasks_local_training_acc, all_tasks_local_training_loss, all_weights_diff
