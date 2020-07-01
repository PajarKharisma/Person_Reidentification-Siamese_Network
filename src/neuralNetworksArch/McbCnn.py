import torch
import torch.nn as nn
from src.utils.CompactBilinearPooling import CountSketch, CompactBilinearPooling

class ConvNetwork_1(nn.Module):
    def __init__(self):
        super(ConvNetwork_1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        output = self.cnn(x)
        return output

class ConvNetwork_2(nn.Module):
    def __init__(self):
        super(ConvNetwork_2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        output = self.cnn(x)
        return output


class PartNetworkOnce(nn.Module):
    def __init__(self):
        super(PartNetworkOnce, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 12))

        self.fc = nn.Sequential(
            nn.Linear(64*4*12, 100),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = self.avgpool(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output

class PartNetwork(nn.Module):
    def __init__(self):
        super(PartNetwork, self).__init__()

        self.net_1 = PartNetworkOnce()
        self.net_2 = PartNetworkOnce()
        self.net_3 = PartNetworkOnce()
        self.net_4 = PartNetworkOnce()

    def forward(self, x):
        x_1 = x[:,:,:8,:]
        x_2 = x[:,:,8:16,:]
        x_3 = x[:,:,16:24,:]
        x_4 = x[:,:,24:32,:]

        out_1 = self.net_1(x_1)
        out_2 = self.net_2(x_2)
        out_3 = self.net_3(x_3)
        out_4 = self.net_4(x_4)

        output = torch.cat([out_1, out_2, out_3, out_4], 1)

        return output


class McbCnn(nn.Module):
    def __init__(self):
        super(McbCnn, self).__init__()
        self.conv_1_up = ConvNetwork_1()
        self.conv_2_up = ConvNetwork_2()

        self.conv_1_bot = ConvNetwork_1()
        self.conv_2_bot = ConvNetwork_2()

        self.part_network_up = PartNetwork()
        self.part_network_bot = PartNetwork()

        self.compact_bil_layer = CompactBilinearPooling(64, 64, 400)

        self.avgpool = nn.AdaptiveAvgPool2d((8, 4))

        self.fc = nn.Sequential(
            nn.Linear(400*8*4, 1000),
            nn.ReLU(inplace=True)
        )

        self._initialize_weights()

    def forward_once(self, x):
        x_up = self.conv_1_up(x)
        x_bot = self.conv_1_bot(x)

        out_part_up = self.part_network_up(x_up)
        out_part_bot = self.part_network_bot(x_bot)

        x_up = self.conv_2_up(x_up)
        x_bot = self.conv_2_bot(x_bot)

        x_up = x_up.permute(0,2,3,1)
        x_bot = x_bot.permute(0,2,3,1)

        output = self.compact_bil_layer(x_up, x_bot)
        output = output.permute(0,3,1,2)

        output = self.avgpool(output)

        output = output.reshape(output.size(0), -1)
        output = self.fc(output)

        output = torch.cat([output, out_part_up, out_part_bot], 1)
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