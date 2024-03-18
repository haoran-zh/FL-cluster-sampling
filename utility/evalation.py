import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset

def evaluation(model, data, batch_size, device, args):
    model.eval()
    # split data into validation set (first 30%) and test set (last 70%)
    # Calculate the number of validation samples
    num_val_samples = int(len(data) * 0.3)
    indices = list(range(len(data)))

    # Shuffle indices if you want to randomize the validation set
    # random.shuffle(indices)

    # Select the indices for the validation and test sets
    val_indices = indices[:num_val_samples]
    test_indices = indices[num_val_samples:]

    if args is None:
        # do nothing, use all data
        pass
    else:
        if args.validation is True:
            data = Subset(data, val_indices)
        else:
            data = Subset(data, test_indices)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    running_loss = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        accuracy = correct/total
        loss = running_loss/total

    return accuracy, loss