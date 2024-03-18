import torch.nn as nn
import torch.nn.functional as F
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=False)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layerss1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layerss2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layerss3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layerss4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion, momentum=None, track_running_stats=False)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layerss1(out)
        out = self.layerss2(out)
        out = self.layerss3(out)
        out = self.layerss4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        #out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PreActResNetMnist(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNetMnist, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layerss1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layerss2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layerss3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layerss4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion, momentum=None, track_running_stats=False)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layerss1(out)
        out = self.layerss2(out)
        out = self.layerss3(out)
        out = self.layerss4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        #out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class MnistMLP(nn.Module):
    def __init__(self, num_classes=10):
        #torch.manual_seed(5)

        super(MnistMLP, self).__init__()
        self.in_size = 28 * 28
        self.hidden_size = 200
        self.out_size = 10
        self.net = nn.Sequential(
            nn.Linear(in_features=self.in_size, out_features=self.hidden_size),
            nn.ReLU(),
            # nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            # nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.out_size),
            #nn.Softmax(dim=2)
            #nn.Softmax(dim=1)
        )

        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def forward(self, batch):
        batch = batch.view(batch.size(0),-1)
        return torch.squeeze(self.net(batch))
    
class MnistCNN(nn.Module):
    def __init__(self,num_classes=10):
        #torch.manual_seed(5)
        super(MnistCNN, self).__init__()
        self.kernel_conv = (3,3)#(5, 5)
        self.kernel_pool = (2, 2)
        self.channel1 = 32
        self.channel2 = 64
        self.conv_out_size = self.channel2*7*7
        self.fc_size = 512
        self.out_size = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channel1, kernel_size=self.kernel_conv, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=self.kernel_pool),
            nn.Conv2d(in_channels=self.channel1, out_channels=self.channel2, kernel_size=self.kernel_conv, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=self.kernel_pool)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.conv_out_size, out_features=self.fc_size),
            nn.Linear(in_features=self.fc_size, out_features=self.out_size),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

        for m in self.modules():
            if type(m) == nn.Conv2d:
                nn.init.kaiming_uniform_(m.weight)
            elif type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def forward(self, batch):
        out1 = self.conv(batch.view(-1, 1, 28, 28)).view(-1, self.conv_out_size)
        return self.fc(out1)

def resnet(num_classes):
    model = PreActResNet(PreActBlock, num_blocks=[2,2,2,2], num_classes=num_classes).to(device)
    model.apply(init_param)
    
    return model

def resnetmnist(num_classes):
    model = PreActResNetMnist(PreActBlock, num_blocks=[2,2,2,2], num_classes=num_classes).to(device)
    model.apply(init_param)
    
    return model

def mnistMLP(num_classes):
    model=MnistMLP(num_classes=num_classes).to(device)
    model.apply(init_param)
    return model

def mnistCNN(num_classes):
    model=MnistCNN(num_classes=num_classes).to(device)
    model.apply(init_param)
    return model



class EMnistCNN(nn.Module):
    def __init__(self, num_classes=47):
        super(EMnistCNN, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)  # 'same' padding
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # Apply layers & activations
        x = F.max_pool2d(F.tanh(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.tanh(self.conv2(x)), kernel_size=2, stride=2)
        x = F.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



def emnistCNN(num_classes, args):
    if args.powerfulCNN is True:
        model = EMnistCNN(num_classes=num_classes).to(device)
        model.apply(init_param)
    else:
        model=MnistCNN(num_classes=num_classes).to(device)
        model.apply(init_param)
    return model

