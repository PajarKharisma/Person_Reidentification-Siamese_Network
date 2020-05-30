import torch.nn as nn

class BasicSiameseNetwork(nn.Module):

    def __init__(self):
        super(BasicSiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16*30*80, 1000),
            nn.ReLU(inplace=True),
            
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500,100),
            nn.ReLU(inplace=True),
            
            nn.Linear(100, 50),
            nn.Sigmoid()
        )
            
        
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2