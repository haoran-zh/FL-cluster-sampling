from .constants import *
from utility.preprocessing import preprocessing
import utility.dataset as dataset
import numpy as np
import torch
import torch.nn as nn

def get_experiment_data(class_set_list):
    tasks_data_info = []
    tasks_data_idx = []
    for i in range(len(task_type)):
        tasks_data_info.append(preprocessing(task_type[i], data_ratio=1.0, args=args)) # 0: trainset, 1: testset, 2: min_data_num, 3: max_data_num 4: input_size, 5: classes_size
        if type_iid[i] =='iid':
            tasks_data_idx.append(dataset.iid(dataset=tasks_data_info[i][0],
                                            min_data_num=tasks_data_info[i][2],
                                            max_data_num=tasks_data_info[i][3],
                                            num_users=num_clients)) # 0: clients_data_idx
        elif type_iid[i] =='noniid':
            tasks_data_idx.append(dataset.detailed_noniid(dataset=tasks_data_info[i][0],
                                min_data_num=tasks_data_info[i][2],
                                max_data_num=tasks_data_info[i][3],
                                class_set=class_set_list,
                                num_users=num_clients))
        return tasks_data_info, tasks_data_idx
    

def get_gradient(weights_this_round, weights_next_round): 
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_this_round[name] - weights_next_round[name]).cpu() for name in weights_this_round}
    return weight_diff

def loss_sampling(m, clients):
    # sample m clients with lowest accuracy
    acc_list = []
    for i in range(len(clients)):
        acc_list.append(clients[i].get_training_accuracy())
    acc_list = np.array(acc_list)
    idx = np.argsort(acc_list)
    idx = idx[:m]
    active_clients = [clients[i] for i in idx]
    return active_clients

def cluster_sampling(active_num, clients, cluster_result):
    len_cluster = [len(cluster) for cluster in cluster_result]
    client_num  = len(clients)
    active_num_cluster = [int(active_num * len_cluster[i] / client_num) for i in range(len(len_cluster))]
    active_clients_list = []
    for i, cluster_clients in enumerate(cluster_result):
        active_clients_list.extend(loss_sampling(m=active_num_cluster[i], clients=cluster_clients))

    return active_clients_list

def flatten_params(state_dict):
    flattened_weights = np.concatenate([param.flatten() for param in state_dict])
    return flattened_weights

def cosine_similarity(state_dict1, state_dict2):
    """Compute the cosine similarity between two model state dictionaries."""
    vec1 = flatten_params(state_dict1)
    vec2 = flatten_params(state_dict2)
    # convert vec1 and vec2 to torch tensors
    vec1 = torch.tensor(vec1)
    vec2 = torch.tensor(vec2)
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    if norm1 > 0 and norm2 > 0:
        return dot_product / (norm1 * norm2)
    else:
        return 0.0

class MnistMLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistMLP, self).__init__()
        self.in_size = 28 * 28
        self.hidden_size = 100
        self.out_size = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_features=self.in_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.out_size),
        )
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def forward(self, batch):
        batch = batch.view(batch.size(0),-1)
        return torch.squeeze(self.net(batch))