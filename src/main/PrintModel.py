import sys
import os
root_dir = os.getcwd()
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import src.neuralNetworksArch.VggArch as vgg
import src.neuralNetworksArch.BasicSiamese as bSiamese
import src.neuralNetworksArch.NasnetMobile as nasnet

# model = vgg.get_model('vgg16', True)
model = nasnet.NASNetAMobile()
# print(model)
summary(model, (3,224,224))