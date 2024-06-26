import torch
import numpy as np
import sys

def iid(dataset, min_data_num, max_data_num, num_users):
    labels = torch.tensor(dataset.targets)
    num_classes = len(labels.unique())
    classes_list = []
    for c in range(num_classes):
        class_list = []
        for idx, label in enumerate(labels):
            if label==c:
                class_list.append(idx)
        classes_list.append(class_list)


    clients_data_idx = []
    classes_data_idx = np.zeros(num_classes, dtype=int)
    for i in range(num_users):
        random_number = np.random.randint(min_data_num[i], max_data_num[i]+1)

        uniform_idx=np.random.randint(0, num_classes)
        client_data_idx = []
        loop_counter = 0
        for _ in range(random_number):
            while(classes_data_idx[uniform_idx] >= len(classes_list[uniform_idx])):
                # len(classes_list[uniform_idx]) how many data points in class=uniform_idx
                # classes_data_idx[uniform_idx]  how many data points in class=uniform_idx have been assigned to clients
                # if all data points in class=uniform_idx have been assigned to clients, then uniform_idx+=1 to jump over this class
                uniform_idx += 1
                if uniform_idx == num_classes:
                    uniform_idx = 0
                loop_counter += 1
                if loop_counter > 3*num_classes:
                    print("data not enough")
                    sys.exit()

            client_data_idx.append(classes_list[uniform_idx][classes_data_idx[uniform_idx]])
            classes_data_idx[uniform_idx] += 1
            uniform_idx += 1
            if uniform_idx == num_classes:
                uniform_idx = 0
        clients_data_idx.append(client_data_idx)
    return clients_data_idx

def noniid(dataset, min_data_num, max_data_num, class_ratio, num_users):
    labels = torch.tensor(dataset.targets)
    num_classes = len(labels.unique())
    classes_list = []
    for c in range(num_classes):
        class_list = []
        for idx, label in enumerate(labels):
            if label==c:
                class_list.append(idx)
        classes_list.append(class_list)


    classes_len_list = []
    for class_list in classes_list:
        classes_len_list.append(len(class_list))

    noniid_class_num = int(class_ratio*num_classes)

    clients_data_idx = []
    classes_data_idx = np.zeros(num_classes, dtype=int)
    clients_label =[]
    for i in range(num_users):
        random_number = np.random.randint(min_data_num[i], max_data_num[i] + 1)
        noniid_labels = []
        # Clone classes_len_list so as not to mutate the original
        temp_classes_len_list = classes_len_list.copy()
        for _ in range(noniid_class_num):
            noniid_label = np.random.choice(np.where(temp_classes_len_list==np.max(temp_classes_len_list))[0],1)[0]
            temp_classes_len_list[noniid_label] = -1
            noniid_labels.append(noniid_label)
        clients_label.append(noniid_labels)
        uniform_idx = np.random.randint(0, noniid_class_num)
        client_data_idx = []
        for _ in range(random_number):
            #print(uniform_idx)
            #print(len(noniid_labels))
            #print(classes_data_idx[noniid_labels[uniform_idx]])
            #print(len(classes_list[noniid_labels[uniform_idx]]))
            while(classes_data_idx[noniid_labels[uniform_idx]] > len(classes_list[noniid_labels[uniform_idx]])):
                uniform_idx += 1
                if uniform_idx == noniid_class_num:
                    uniform_idx = 0
            client_data_idx.append(classes_list[noniid_labels[uniform_idx]][classes_data_idx[noniid_labels[uniform_idx]]])
            classes_data_idx[noniid_labels[uniform_idx]] += 1
            classes_len_list[noniid_labels[uniform_idx]] -= 1
            uniform_idx += 1
            if uniform_idx == noniid_class_num:
                uniform_idx = 0
        clients_data_idx.append(client_data_idx)
    return clients_data_idx, clients_label


def detailed_noniid(dataset, min_data_num, max_data_num, class_set, num_users):
    # class_set: example [[1,2],[3,4],[5,6],[1,9]]
    # directly define possible local class pairs for all clients
    labels = torch.tensor(dataset.targets)
    num_classes = len(labels.unique())  # we need all possible class pairs
    
    classes_list = []
    for c in range(num_classes):
        class_list = []
        for idx, label in enumerate(labels):
            if label==c:
                class_list.append(idx)
        classes_list.append(class_list)
    # record all data index for each class

    classes_len_list = []
    for class_list in classes_list:
        classes_len_list.append(len(class_list))

    clients_data_idx = []
    clients_label = []
    for i in range(num_users):
        # random choose a class pair from class_set
        class_pair_index = np.random.choice(np.arange(len(class_set)), 1)[0]
        class_pair = class_set[class_pair_index]
        clients_label.append(class_pair)
        # decide how many data points for this client
        random_number = np.random.randint(min_data_num[i], max_data_num[i] + 1)
        # get the data index for each class in the class pair
        client_data_idx = []
        for _ in range(random_number):
            # randomly choose a class from the class pair
            this_class = np.random.choice(class_pair, 1)[0]
            # randomly choose a data point from this class
            this_index = np.random.randint(0, len(classes_list[this_class]))
            client_data_idx.append(classes_list[this_class][this_index])
            # remove this data point from the class list
            classes_list[this_class].pop(this_index)
        clients_data_idx.append(client_data_idx)
    
    return clients_data_idx, clients_label