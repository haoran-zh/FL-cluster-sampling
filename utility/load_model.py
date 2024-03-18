import utility.model_list as model_list
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(name_data, num_classes, args):
    if name_data =='cifar10':
        model = model_list.resnet(num_classes=num_classes)#.to(device)
        return model
    elif name_data =='mnist':
        #model = model_list.resnetmnist(num_classes=num_classes)
        #model = model_list.mnistMLP(num_classes=num_classes)#.to(device)
        model = model_list.mnistCNN(num_classes=num_classes)#.to(device)
        return model
    
    elif name_data=='fashion_mnist':
        model = model_list.mnistCNN(num_classes=num_classes)#.to(device)
        return model
    elif name_data=='emnist':
        model = model_list.emnistCNN(num_classes=num_classes, args=args)
        return model

