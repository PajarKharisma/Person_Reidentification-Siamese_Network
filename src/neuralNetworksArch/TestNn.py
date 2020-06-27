import torch
import torch.nn as nn

class TestNN(nn.Module):

    def __init__(self):
        super(TestNN, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((40, 15))

        self.fc1 = nn.Sequential(
            nn.Linear(16*40*15, 1000),
            nn.ReLU(inplace=True),
            
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500,100),
            nn.ReLU(inplace=True),
            
            nn.Linear(100, 50),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 2),
            nn.Sigmoid()
        )

        self._initialize_weights()
        
    def forward_once(self, x):
        output = self.cnn1(x)
        output = self.avgpool(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2, input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        print('Output1 :', output1.size())
        if input3 == None:
            input_cat = torch.cat((output1, output2, output2, output1), 0)
            print(input_cat.size())
            final_output = self.fc2(input_cat)
            return final_output
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