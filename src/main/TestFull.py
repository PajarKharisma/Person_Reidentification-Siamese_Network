import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torchvision.utils
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import copy
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import src.neuralNetworksArch.BstCnn as bst

import src.utils.DatasetLoader as dsetLoader
import src.utils.Checkpoint as ckp
import src.utils.Metrics as metrics

from src.config.Path import *
from src.config.Param import *

trans = transforms.Compose([transforms.ToTensor()])
contrastive_dataset = dsetLoader.ContrastiveDataset(
    csv_path=Path.test_csv,
    images_path=Path.test_images,
    transform=trans,
    resize=Param.input_size
)

dataset = DataLoader(contrastive_dataset, batch_size=Param.train_batch_size, shuffle=True)

model  = bst.BstCnn()
checkpoint = ckp.load_checkpoint(load_dir=Path.load_model)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(Param.device)
model.eval()

y_true = torch.tensor([], dtype=torch.float).to(Param.device)
y_scores = torch.tensor([], dtype=torch.float).to(Param.device)

with torch.no_grad():
    for i, data in enumerate(dataset):
        x1, x2, x3 = data
        x1 = Variable(x1.to(Param.device))
        x2 = Variable(x2.to(Param.device))
        x3 = Variable(x3.to(Param.device))

        if Param.data_type == 'PAIR':
            output1, output2 = model(x1, x2)
            output3 = x3

            dist = metrics.get_distances(output1, output2)
            y_true = torch.cat((y_true, output3))
            y_scores = torch.cat((y_scores, dist))

        else:
            output1, output2, output3 = model(x1, x2, x3)

y_true = y_true.flatten().detach().cpu().numpy()
y_scores = y_scores.flatten().detach().cpu().numpy()
max_dist = checkpoint['max_dist']
y_scores = metrics.normalize_dist(y_scores, max_dist)

best_thresh = checkpoint['threshold']
y_pred = metrics.distance_to_class([y_scores], best_thresh)

acc = accuracy_score(y_true, y_pred)
print('Akurasi : {}'.format(acc))