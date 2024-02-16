import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetLin(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNetLin, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2), #inChannels, outChannels, kernel_size, stride, padding
            #nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            #nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
        )
        self.fc = nn.Linear(25088, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
class ConvNetReLU(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNetReLU, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2), #inChannels, outChannels, kernel_size, stride, padding
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(25088, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out