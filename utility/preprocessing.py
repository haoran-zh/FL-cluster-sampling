import torchvision.transforms as transforms
import torchvision

def preprocessing(name_data, data_ratio, args):
    unbalance = args.unbalance
    clients_num = args.num_clients
    unbalanced_clients = int(unbalance[0] * clients_num)
    unbalanced_data = unbalance[1]

    if name_data == 'cifar10':
        # Load the CIFAR10 training and test datasets
        transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./utility/dataset', train=True, download=True, transform=transforms_train)
        testset = torchvision.datasets.CIFAR10(root='./utility/dataset', train=False, download=True, transform=transforms_test)
        input_size = 32
        classes_size = 10
        min_data_num = int(data_ratio*400)
        max_data_num = int(data_ratio*500)

    elif name_data == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])        

        trainset = torchvision.datasets.MNIST('./utility/dataset', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST('./utility/dataset', train=False, download=True, transform=transform_test)
        input_size = 28
        classes_size = 10
        min_data_num = int(data_ratio*500)
        max_data_num = int(data_ratio*600)
        
    elif name_data=='fashion_mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        
        trainset=torchvision.datasets.FashionMNIST('./utility/dataset', train=True, download=True, transform=transform_train)
        testset=torchvision.datasets.FashionMNIST('./utility/dataset', train=False, download=True, transform=transform_test)
        
        input_size=28
        classes_size = 10
        min_data_num = int(data_ratio*400)
        max_data_num = int(data_ratio*500)

    elif name_data == 'emnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.EMNIST('./utility/dataset', train=True, download=True,
                                                     transform=transform_train, split='balanced')
        testset = torchvision.datasets.EMNIST('./utility/dataset', train=False, download=True,
                                                    transform=transform_test, split='balanced')

        input_size = 28
        classes_size = 47 # excluding capital letters that look similar to their lowercase counterparts
        min_data_num = int(data_ratio*400)
        max_data_num = int(data_ratio*500)

    max_data_list = []
    min_data_list = []
    unbalanced_max = max_data_num * unbalanced_data
    unbalanced_min = min_data_num * unbalanced_data
    for i in range(clients_num):
        if i < unbalanced_clients:
            max_data_list.append(unbalanced_max)
            min_data_list.append(unbalanced_min)
        else:
            max_data_list.append(max_data_num)
            min_data_list.append(min_data_num)

        
    return trainset, testset, min_data_list, max_data_list, input_size, classes_size
