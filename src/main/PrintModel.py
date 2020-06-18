import sys
import os
root_dir = os.getcwd()
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from src.config.Param import *

import src.neuralNetworksArch.VggArch as vgg
import src.neuralNetworksArch.BasicSiamese as bSiamese
import src.neuralNetworksArch.NasnetMobile as nasnet
import src.neuralNetworksArch.AdaptiveSpatialFeature as asf
import src.neuralNetworksArch.BstCnn as btsCnn
import src.neuralNetworksArch.McbCnn as mcbCnn

model = mcbCnn.McbCnn()
model = model.to(Param.device)
# print(model)
summary(model, [(3, 60, 160), (3, 60, 160)])