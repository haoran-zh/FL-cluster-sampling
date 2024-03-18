import numpy as np
import torch
import utility.dataset as dataset
from utility.preprocessing import preprocessing
from utility.load_model import load_model
from utility.training import training, training_all
from utility.evalation import evaluation
from utility.aggregation import federated, federated_prob
from utility.taskallocation import get_task_idx, get_task_id_RR
import random
import time
import sys
import math
import os
from torch.utils.data import Subset
from tqdm import tqdm
import argparse
from utility.parser import ParserArgs
import utility.optimal_sampling as optimal_sampling

if __name__=="__main__":
    parser = ParserArgs()
    args = parser.get_args()
    exp_num = args.exp_num
    random_seed = args.seed  # default 13
    C = args.C  # default 1 #0.2#0.1
    num_clients = args.num_clients  # default 30 #100
    numUsersSel = C * num_clients

    normalization = 'accuracy'  # accuracy
    num_round = args.round_num  # 100#200
    print('num_round', num_round)
    local_epochs = args.local_epochs
    print(local_epochs, 'local_epochs')
    batch_size = 500
    EMNIST_powerfulCNN = args.powerfulCNN
    type_iid = args.iid_type  # 'iid', 'noniid'
    class_ratio = args.class_ratio  # non iid only
    beta = args.alpha  # default 3
    task_type = args.task_type
    task_number = len(task_type)
    data_ratio = args.data_ratio
    # set record name
    iid_str = ''.join([x[0] for x in type_iid])
    task_str = ''.join([x[0] for x in task_type])
    iid_filename = iid_str + task_str + '_a' + str(beta) + 'P' if EMNIST_powerfulCNN \
        else iid_str + task_str + '_a' + str(beta) + 'l'
    tasks_weight = np.ones(len(task_type)) / len(task_type)

    folder_name = str(task_number)+"task_"+iid_str+"_"+args.notes
    if not os.path.exists('./result/'+folder_name):
        os.makedirs('./result/'+folder_name)
    else:
        if args.insist is True:
            # go ahead
            print('folder exists but still go on!')
        else:
            print('folder exists!')
            sys.exit()

    print('task number', task_number)
    print('folder name', folder_name)
    print('exp num', exp_num)
    print('use powerful CNN?', EMNIST_powerfulCNN)
    print('iid type', type_iid)
    print('task type', task_type)

    for exp in range(0,exp_num):
        algorithm_name_vec = args.algo_type
        aggregation_dict = {'bayesian':'pkOverSumPk',
                            'proposed':'pkOverSumPk',
                            'random':'numUsersInv',
                            'round_robin':'numUsersInv'}

        aggregation_mtd_vec=[aggregation_dict[algo] for algo in algorithm_name_vec]

        for algo in range(len(algorithm_name_vec)):
            algorithm_name = algorithm_name_vec[algo]
            aggregation_mtd = aggregation_mtd_vec[algo]
            print('exp', exp, 'algo', algorithm_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#'cuda:0'

            #for round robin
            rr_taskAlloc=np.zeros(num_clients)
            #rr_taskAlloc[0:int(len(rr_taskAlloc)/2)]=1
            first_ind=0
            for i in range(len(task_type)):
                rr_taskAlloc[first_ind:math.floor((i+1)*num_clients/len(task_type))]=i
                first_ind=math.floor((i+1)*num_clients/len(task_type))
            firstIndRR=0
            #rr_clients=np.arange(num_clients)
            #print(rr_taskAlloc, 'rr_taskAlloc')
            clients_task=0

            file =  open('./result/'+folder_name+'/Algorithm_'+algorithm_name+'_normalization_'+normalization+'_type_'+iid_filename+'_seed_'+str(random_seed)+''+'.txt', 'w')


            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(random_seed)

            global_models = []
            local_results = []
            global_results = []
            tasks_data_info = []
            tasks_data_idx = []
            globalAccResults=np.zeros((len(task_type),num_round))
            globalLossResults=np.zeros((len(task_type),num_round))

            TaskAllocCounter=np.zeros((len(task_type),num_round))

            # initialize global model
            for i in range(len(task_type)):
                tasks_data_info.append(preprocessing(task_type[i], data_ratio, args)) # 0: trainset, 1: testset, 2: min_data_num, 3: max_data_num 4: input_size, 5: classes_size
                if type_iid[i] =='iid':
                    tasks_data_idx.append(dataset.iid(dataset=tasks_data_info[i][0],
                                                    min_data_num=tasks_data_info[i][2],
                                                    max_data_num=tasks_data_info[i][3],
                                                    num_users=num_clients)) # 0: clients_data_idx
                elif type_iid[i] =='noniid':
                    tasks_data_idx.append(dataset.noniid(dataset=tasks_data_info[i][0],
                                        min_data_num=tasks_data_info[i][2],
                                        max_data_num=tasks_data_info[i][3],
                                        class_ratio=class_ratio[i],
                                        num_users=num_clients)) # 0: clients_data_idx 1: clients_label
                global_models.append(load_model(name_data=task_type[i], num_classes=tasks_data_info[i][5], args=args).to(device))
                local_results.append([-1,-1])
                global_results.append([-1,-1])


            # record all client data num
            all_data_num = []
            for task_idx in range(len(task_type)):
                local_data_num = []
                for client_idx in range(num_clients):
                    if type_iid[task_idx] == 'iid':
                        local_data_num.append(len(tasks_data_idx[task_idx][client_idx]))
                    if type_iid[task_idx] == 'noniid':
                        local_data_num.append(len(tasks_data_idx[task_idx][0][client_idx]))
                all_data_num.append(local_data_num)

            # allocation_initialization !!
            # NEW: dec 6 2023
            all_clients = list(range(0, num_clients))
            chosen_clients = random.sample(all_clients, int(numUsersSel))
            # we first randomly allocate.
            clients_task = np.random.randint(0, len(task_type), int(num_clients * C), dtype=int)

            # allocation_initialization
            allocation_history_list = []  # record history of allocation for bayesian
            allocation_history_list.append(clients_task)

            # if bayesian, we need to initialize P_task_client_bayesian
            if algorithm_name == 'bayesian':
                P_task_client_bayesian = np.ones((num_clients, task_number)) / task_number

            for round in tqdm(range(num_round)):
                #print(round)
                #print(f"Round [{round+1}/{num_round}]")
                print(f"Round[ {round+1}/{num_round} ]",file=file)
                #print("Allocated Tasks:", clients_task)
                # training
                if args.optimal_sampling is True:
                    all_tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, all_weights_diff = training_all(
                                                                                        tasks_data_info=tasks_data_info, tasks_data_idx=tasks_data_idx,
                                                                                        global_models=global_models, chosen_clients=chosen_clients,
                                                                                        task_type=task_type, clients_task=None,
                                                                                        local_epochs=local_epochs, batch_size=batch_size, classes_size=tasks_data_info,
                                                                                        type_iid=type_iid, device=device, args=args)
                    # optimal sampling
                    if args.alpha_loss is True:
                        all_weights_diff_power = optimal_sampling.power_gradient_norm(all_weights_diff, tasks_local_training_loss, args, all_data_num)
                        clients_task, p_dict, chosen_clients = optimal_sampling.get_optimal_sampling(chosen_clients,
                                                                                                     clients_task,
                                                                                                     all_data_num,
                                                                                                     all_weights_diff_power, args)
                    else:
                        clients_task, p_dict, chosen_clients = optimal_sampling.get_optimal_sampling_cvx(chosen_clients,
                                                                                                         clients_task,
                                                                                                         all_data_num,
                                                                                                         all_weights_diff)

                    # optimal sampling needs to be moved after we get local_data_nums
                else:
                    # if args.approx_optimal, then get all local loss and acc, update chosen_clients and clients_task
                    if args.approx_optimal is True:
                        # 1 get all local loss
                        localAcc = np.zeros((task_number, num_clients))
                        localLoss = np.zeros((task_number, num_clients))
                        for cl in range(num_clients):
                            for task in range(task_number):
                                if type_iid[task] == 'iid':
                                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][
                                        cl])  # or iid_partition depending on your choice
                                if type_iid[task] == 'noniid':
                                    client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][0][
                                        cl])  # or iid_partition depending on your choice
                                accu, loss = evaluation(model=global_models[task], data=client_data,
                                                        batch_size=batch_size, device=device, args=None)  # use all data
                                localAcc[task, cl] = accu
                                localLoss[task, cl] = loss
                        # use loss to replace gradient norm
                        if args.alpha_loss is True:
                            all_weights_diff_power = optimal_sampling.power_gradient_norm(localLoss,
                                                                                  localLoss, args,
                                                                                  all_data_num)
                            clients_task, p_dict, chosen_clients = optimal_sampling.get_optimal_sampling(chosen_clients,
                                                                                                     clients_task,
                                                                                                     all_data_num,
                                                                                                     all_weights_diff_power, args)
                        else:
                            clients_task, p_dict, chosen_clients = optimal_sampling.get_optimal_sampling_cvx(chosen_clients,
                                                                                                         clients_task,
                                                                                                         all_data_num,
                                                                                                         localLoss)




                    tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, all_weights_diff, loss_bf_agg = training(tasks_data_info=tasks_data_info, tasks_data_idx=tasks_data_idx,
                                                                                                   global_models=global_models, chosen_clients=chosen_clients,
                                                                                                   task_type=task_type, clients_task=clients_task,
                                                                                                   local_epochs=local_epochs, batch_size = batch_size, classes_size = tasks_data_info,
                                                                                                    type_iid=type_iid, device=device, args=args)



                if args.optimal_sampling is True:
                    # remember to process local_loss
                    temp_global_results = []
                    for task_idx in range(len(task_type)):
                        this_task_gradients_list = []
                        # get local_weights for this task
                        for client_idx in range(len(chosen_clients)):
                            if clients_task[client_idx] == task_idx:
                                this_task_gradients_list.append(all_tasks_gradients_list[task_idx][chosen_clients[client_idx]])
                        assert len(this_task_gradients_list) == len(p_dict[task_idx])
                        # aggregation
                        if (len(this_task_gradients_list) != 0):
                            if args.cpumodel is True:
                                global_models[task_idx].to('cpu')
                            global_models[task_idx].load_state_dict(
                                federated_prob(global_weights =global_models[task_idx], models_gradient_dict=this_task_gradients_list, local_data_num=all_data_num[task_idx],
                                          p_list=p_dict[task_idx], args=args, chosen_clients=chosen_clients, tasks_local_training_loss=tasks_local_training_loss[task_idx]))
                            if args.cpumodel is True:
                                global_models[task_idx].to(device)
                            temp_global_results.append(
                                evaluation(model=global_models[task_idx], data=tasks_data_info[task_idx][1],
                                           batch_size=batch_size, device=device, args=args))
                            # print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}")
                            print('p_list', p_dict[task_idx], file=file)
                            #print('p_list', p_dict[task_idx])
                            print(
                                f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}",
                                file=file)
                            #print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}")
                        else:
                            temp_global_results.append(global_results[task_idx])
                            # print(f"Task[{task_idx}]: Global not changed")
                            print(f"Task[{task_idx}]: Global not changed", file=file)

                    global_accs = []
                    for task_idx in range(len(task_type)):
                        global_accs.append(temp_global_results[task_idx][0])
                else:
                    temp_global_results = []
                    for task_idx in range(len(task_type)):
                        temp_local_weights = []
                        temp_local_gradients = []
                        temp_local_P = []
                        temp_local_data_num = []
                        local_data_nums = []
                        for clients_idx, local_gradients in enumerate(tasks_gradients_list):
                            if clients_task[clients_idx] == task_idx:
                                temp_local_gradients.append(local_gradients)
                                if algorithm_name == 'bayesian':
                                    temp_local_P.append(P_task_client_bayesian[chosen_clients[clients_idx]][task_idx])
                                    # P_{i,s}=P(s|i)*P(i)
                                elif args.approx_optimal is True:
                                    temp_local_P = p_dict[task_idx]
                                else:
                                    # print('uniform distribution')
                                    p = C/task_number
                                    temp_local_P.append(p)
                                # do not need to collect local_data num, just use all_local_data_num
                        #print('task, local data nums', task_idx, local_data_nums)
                        # aggregation
                        if ((len(temp_local_weights)+len(temp_local_gradients)) !=0):
                            if args.cpumodel is True:
                                global_models[task_idx].to('cpu')
                            # 抽取P_task_client_bayesian based on chosen_clients and clients_task, define a function to do that
                            global_models[task_idx].load_state_dict(
                                federated_prob(global_weights=global_models[task_idx],
                                               models_gradient_dict=temp_local_gradients,
                                               local_data_num=all_data_num[task_idx],
                                               p_list=temp_local_P, args=args, chosen_clients=chosen_clients, tasks_local_training_loss=tasks_local_training_loss))
                            print('p_list', temp_local_P, file=file)
                            #print('p_list', temp_local_P)

                            if args.cpumodel is True:
                                global_models[task_idx].to(device)
                            temp_global_results.append(evaluation(model = global_models[task_idx], data = tasks_data_info[task_idx][1], batch_size = batch_size, device = device, args=args))
                            #print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}")
                            print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}",file=file)
                            #print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}")
                        else:
                            temp_global_results.append(global_results[task_idx])
                            #print(f"Task[{task_idx}]: Global not changed")
                            print(f"Task[{task_idx}]: Global not changed",file=file)

                    global_accs = []
                    for task_idx in range(len(task_type)):
                        global_accs.append(temp_global_results[task_idx][0])
                # NEW: dec 6 2023

                all_clients = list(range(0, num_clients))
                chosen_clients = random.sample(all_clients, int(numUsersSel))  # for next round.

                if algorithm_name == 'round_robin':
                    clients_task, rr_chosen_clients, firstIndRR, rr_taskAlloc = get_task_id_RR(num_tasks=len(task_type),
                                                                        totNumCl=num_clients,
                                                                        num_clients=int(num_clients * C),
                                                                        algorithm_name=algorithm_name,
                                                                       normalization=normalization,
                                                                       tasks_weight=tasks_weight,
                                                                       global_accs=global_accs,
                                                                       beta=beta,
                                                                       firstIndRR=firstIndRR,
                                                                       rr_taskAlloc=rr_taskAlloc,
                                                                       # rr_chosen_clients=rr_chosen_clients)
                                                                       # NEW: dec 6 2023
                                                                       rr_chosen_clients=chosen_clients)
                else:
                    clients_task = get_task_idx(num_tasks=len(task_type), num_clients=int(num_clients * C),
                                                algorithm_name=algorithm_name,
                                                normalization=normalization,
                                                tasks_weight=tasks_weight, global_accs=global_accs, beta=beta,
                                                # NEW: dec 6 2023
                                                chosen_clients=chosen_clients,
                                                allocation_history=allocation_history_list,
                                                args=args)

                TaskAllocCounter[:, round] = np.bincount(np.array(clients_task).astype(np.int64), minlength=len(task_type))
                #print("alloc", TaskAllocCounter[:, round])

                #local_results = temp_local_results
                global_results = temp_global_results

                globalAccResults[:,round]=np.array(temp_global_results)[:,0]
                #localAccResults[:,round]=np.array(temp_local_results)[:,0]
                globalLossResults[:,round]=np.array(temp_global_results)[:,1]
                #localLossResults[:,round]=np.array(temp_local_results)[:,1]

            # record global model performance on local client data
            # only record the last round performance
            localAcc = np.zeros((task_number, num_clients))
            localLoss = np.zeros((task_number, num_clients))
            for cl in range(num_clients):
                for task in range(task_number):
                    if type_iid[task] == 'iid':
                        client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][
                            cl])  # or iid_partition depending on your choice
                    if type_iid[task] == 'noniid':
                        client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][0][
                            cl])  # or iid_partition depending on your choice
                    accu, loss = evaluation(model=global_models[task], data=client_data,
                                            batch_size=batch_size, device=device, args=None)  # use all data
                    localAcc[task, cl] = accu
                    localLoss[task, cl] = loss



            filename='mcf_i_globalAcc_exp{}_algo{}.npy'.format(exp,algo)
            np.save('./result/'+folder_name+'/'+ filename, globalAccResults)
            filename='mcf_i_globalLoss_exp{}_algo{}.npy'.format(exp,algo)
            np.save('./result/'+folder_name+'/'+ filename, globalLossResults)

            filename='localAcc_exp{}_algo{}.npy'.format(exp,algo)
            np.save('./result/'+folder_name+'/'+ filename, localAcc)
            filename='localLoss_exp{}_algo{}.npy'.format(exp,algo)
            np.save('./result/'+folder_name+'/'+ filename, localLoss)

            filename = 'allocCounter_{}.npy'.format(algo)
            np.save('./result/'+folder_name+'/'+filename, TaskAllocCounter)

            print(f'Finished Training, lr:{args.lr}, Global acc:{globalAccResults[:, -1]}')
            print(f'Finished Training, lr:{args.lr}, Global acc:{globalAccResults[:, -1]}', file=file)
            file.close()