from .client import Client
from .utils.constants import *
from .utils.utils import *
from .cluster import Cluster
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class Server:
    def __init__(self, tasks_data_info, tasks_data_idx, batch_size=128, tasks_index=0, cluster_requirements=None):
        #global tasks_data_info, tasks_data_idx
        self.tasks_data_info = tasks_data_info
        self.tasks_data_idx = tasks_data_idx
        self.batch_size = batch_size
        self.tasks_index = tasks_index # 0 is non-iid
        self.test_data = self.tasks_data_info[self.tasks_index][1] # set test dataset
        self.loss_list = []
        self.cluster_requirements = cluster_requirements
        
        self.global_model = MnistMLP()
        self.device = torch.device("cpu")
        #self.training()
        #self.get_trained_weights()
        # load global model

    def aggregation(self):
        # aggregate weights
        global_weights_state_dict = self.global_model.state_dict()
        global_keys = list(global_weights_state_dict.keys())
        for key in global_keys:
            global_weights_state_dict[key] = torch.zeros_like(global_weights_state_dict[key])

        fraction_sum = 0
        for c in self.active_clients:
            fraction_sum += c.data_fraction

        for c in self.active_clients:
            for key in global_weights_state_dict.keys():
                global_weights_state_dict[key] += c.data_fraction / fraction_sum * c.model.state_dict()[key]

        # update global model
        self.global_model.load_state_dict(global_weights_state_dict)
        
    def sampling(self, clients, method='random'):
        # global active_rate
        active_num = int(active_rate * len(clients))
        if method == 'random':
            self.active_clients = random.sample(clients, active_num)
        elif method == 'IS':
            self.active_clients = loss_sampling(m=active_num, clients=clients)
        elif method == 'cluster':
            assert self.cluster_requirements is not None, "Cluster requirements not set."
            clusters = Cluster(clients)
            clusters.cluster(*self.cluster_requirements)
            if (self.current_round+1) % 10 == 0:
                clusters.plot()
            cluster_result = clusters.get_result()
            self.active_clients = cluster_sampling(active_num, clients, cluster_result)
    
    def get_global_loss(self):
        testloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        self.global_model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(testloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss += criterion(output, target)
            loss /= len(testloader)
        # print('global loss: ', loss.item())
        return loss.item()
    
    def FL_training(self, num_clients=20, rounds=10, sampling_method='random', experiment_num=1):
        # define clients
        clients = [Client(self.tasks_data_info, self.tasks_data_idx, i) for i in range(num_clients)]
        experiment_loss_lists = []
        for exp in range(experiment_num):
            for r in tqdm(range(rounds)):
                self.current_round = r
                # set local models weights to current global
                for c in clients:
                    c.update_to_global_weights(self.global_model)
                # sampling with current global model
                self.sampling(clients, method=sampling_method)
                # train clients
                for c in self.active_clients:
                    c.training(epochs=5)
                # aggregate weights
                self.aggregation()
                self.loss_list.append(self.get_global_loss())
            loss_list = self.loss_list.copy()
            experiment_loss_lists.append(loss_list)
            self.loss_list = []
        # average loss over experiments
        avg_loss_list = np.mean(np.array(experiment_loss_lists), axis=0)
        self.loss_list = avg_loss_list.tolist()

    
    def plot_loss(self):
        plt.plot(self.loss_list)
        plt.show()