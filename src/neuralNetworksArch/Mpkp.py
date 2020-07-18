import torch
import torch.nn as nn

class MpkpCnn(nn.Module):
    def __init__(self):
        super(MpkpCnn, self).__init__()

        self.conv5x5_1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 3, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3)
        )

        self.conv3x3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 3, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3)
        )

        self.conv5x5_2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(9, 12, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12)
        )

        self.conv3x3_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(9, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12)
        )

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(9, 12, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12)
        )

        self.conv3x3_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(36, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((16, 8)) # full images

        self.fc = nn.Sequential(
            nn.Linear(128*16*8, 4096), # full images
            nn.ReLU(inplace=True),

            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512)
        )

        self._initialize_weights()

    def forward_once(self, x):
        out_5x5 = self.conv5x5_1(x)
        out_3x3 = self.conv3x3_1(x)
        output = torch.cat([out_5x5, out_3x3, x], 1)
        output = self.maxpool(output)
        
        out_5x5 = self.conv5x5_2(output)
        out_3x3 = self.conv3x3_2(output)
        out_1x1 = self.conv1x1_2(output)
        output = torch.cat([out_5x5, out_3x3, out_1x1], 1)
        output = self.maxpool(output)

        out_3x3 = self.conv3x3_3(output)
        out_1x1 = self.conv1x1_3(output)
        output = torch.cat([out_3x3, out_1x1], 1)
        output = self.avgpool(output)

        output = output.reshape(output.size(0), -1)
        output = self.fc(output)

        return output

    def forward(self, input1, input2, input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if input3 == None:
            return output1, output2
        else:
            output3 = self.forward_once(input3)
            return output1, output2, output3

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
