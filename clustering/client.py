from .utils.constants import *
from .utils.utils import *
from torch.utils.data import Subset, DataLoader

class Client:
    def __init__(self, tasks_data_info, tasks_data_idx, client_index, batch_size=256, tasks_index=0):
        #global tasks_data_info, tasks_data_idx
        self.client_index = client_index
        self.tasks_data_info = tasks_data_info
        self.tasks_data_idx = tasks_data_idx
        self.batch_size = batch_size
        self.tasks_index = tasks_index # 0 is non-iid
        self.gradient = None
        self.set_data()
        self.model = MnistMLP()
        self.device = torch.device("cpu")
        self.training(epochs=10)
        # print(f"Client {client_index}; labels {self.non_iid_labels}; accuracy {np.round(self.get_training_accuracy(), 3)}")
        self.get_trained_weights()

        local_data_num = []
        for client_idx in range(num_clients):
            local_data_num.append(len(tasks_data_idx[self.tasks_index][0][client_idx]))
        self.data_fraction = len(self.loader.dataset) / sum(local_data_num)
        
        
    def set_data(self):
        if type_iid[self.tasks_index] == 'iid':
            client_data = Subset(self.tasks_data_info[self.tasks_index][0], 
                                 self.tasks_data_idx[self.tasks_index][self.client_index])  # or iid_partition depending on your choice
        elif type_iid[self.tasks_index] == 'noniid':
            client_data = Subset(self.tasks_data_info[self.tasks_index][0], 
                                 self.tasks_data_idx[self.tasks_index][0][self.client_index])  # or iid_partition depending on your choice
            self.non_iid_labels = self.tasks_data_idx[self.tasks_index][1][self.client_index]
        self.loader = DataLoader(client_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    
    def training(self, epochs=10):
        loader = self.loader
        self.model.to(self.device)
        previous_local_state_dict = self.model.state_dict().copy()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        client_label = self.tasks_data_idx[self.tasks_index][1][self.client_index]


        for _ in range(epochs):
            self.model.train()
            loss = 0
            for i, (data, target) in enumerate(loader):
                optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                label_mask = torch.zeros(self.tasks_data_info[self.tasks_index][5], device=output.device)
                label_mask[client_label] = 1
                output = output.masked_fill(label_mask == 0, 0)
                     
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss += loss.item()
        self.gradients = get_gradient(previous_local_state_dict, self.model.state_dict())
        return self.model
    
    def get_training_accuracy(self):
        self.model.to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total
    
    def get_trained_weights(self):
        # get weights
        self.weights = []
        for param in self.model.parameters():
            self.weights.append(param.data.cpu().numpy())
        self.flattened_weights = np.concatenate([w.flatten() for w in self.weights])
        # get graidents
        self.gradients = []
        for p in self.model.parameters():
            self.gradients.append(p.grad.data.cpu().numpy())
        self.flattened_gradients = np.concatenate([g.flatten() for g in self.gradients])
        # get gradident norms
        self.gradient_norms = [np.linalg.norm(g) for g in self.gradients]

    def update_to_global_weights(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        
    def get_local_weights(self):
        return self.model.state_dict()