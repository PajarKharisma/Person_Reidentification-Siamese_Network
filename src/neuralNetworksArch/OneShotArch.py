import torch.nn as nn

class OneShotArch(nn.Module):
    def __init__(self, init_weights=True):
        super(OneShotArch, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((15, 40))

        self.fc1 = nn.Sequential(
                nn.Linear(256*15*40, 4096),
                nn.Sigmoid()
            )

        self.fc2 = nn.Sequential(
                nn.Linear(4096, 1),
                nn.Sigmoid()
            )

        self._initialize_weights()

    def forward_once(self, x):
        output = self.cnn(x)
        otuput = self.avgpool(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def fordward_last(self, input):
        output = self.fc2(input)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
