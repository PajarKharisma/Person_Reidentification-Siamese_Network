import torch
import torch.nn as nn

class PartNetwork(nn.Module):
    def __init__(self):
        super(PartNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 11))

        self.fc = nn.Sequential(
            nn.Linear(64*4*11, 100),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = self.avgpool(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output