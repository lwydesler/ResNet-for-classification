import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(outchannel),
                                          nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential( nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 1024, 2, stride=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(1024, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1) #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, 2)
        out = self.layer5(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #out = F.dropout(out, 0.3)
        out = F.log_softmax(self.fc2(out), dim=1)

        return out

def ResNet18():

    return ResNet(ResidualBlock)

class ResNet1(nn.Module):
    def __init__(self):
        super(ResNet1, self).__init__()
        self.tmp_dict = {}

        self.conv1 = nn.Conv2d(3, 12, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 36, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 108, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(108)
        self.conv4 = nn.Conv2d(108, 216, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(216)
        self.fc1 = nn.Linear(1944, 648)#4608
        self.fc2 = nn.Linear(648, 216)
        self.fc3 = nn.Linear(216, 12)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 50 25

        x = self.conv2(x)  #
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 25 17

        x = self.conv3(x)# 14-4 = 10 12 - 4 = 8
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 16 8

        x = self.conv4(x)  # 14-4 = 10 12 - 4 = 8
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  #  8   4

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, 0.3)



        x = F.log_softmax(self.fc3(x), dim=1)

        return x