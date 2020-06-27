import torch
import torch.nn.functional as F

import copy
import numpy as np
from sklearn.metrics import accuracy_score

from src.config.Param import *

def distance_to_class(distances, threshold=0.5, margin=1.0, data_type='PAIR'):
    if data_type == 'PAIR':
        distances = distances[0].flatten().detach().cpu().numpy()
        distances_norm = [abs((1 / (1 + d)) - 1) for d in distances]
        y = [0.0 if d <= threshold else 1.0 for d in distances_norm]
    else:
        dist_p = distances[0].flatten().detach().cpu().numpy()
        dist_n = distances[1].flatten().detach().cpu().numpy()
        y = []
        for i in range(len(dist_p)):
            dist = 0.0 if dist_p[i] + margin <= dist_n[i] else 1.0
            y.append(dist)
    return np.array(y)

def get_distances(x1, x2):
    return F.pairwise_distance(x1, x2, keepdim = True)

def get_acc(x1, x2, x3, threshold=0.5, data_type='PAIR'):
    if data_type == 'PAIR':   
        y_true = x3.flatten().cpu().numpy()
        distances = [get_distances(x1, x2)]
        y_pred = distance_to_class(distances=distances, threshold=threshold)
    else:
        y_true = np.full((len(x1)), 0.0)
        dist_a = get_distances(x1,x2)
        dist_b = get_distances(x1,x3)
        distances = [dist_a, dist_b]
        y_pred = distance_to_class(distances=distances, data_type=data_type)
    
    return accuracy_score(y_true, y_pred)

def get_val_metrics(model, dataset, loss_func, threshold=0.5, data_type='PAIR'):
    model.eval()
    model.zero_grad()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x1, x2, x3 = data
            x1 = x1.to(Param.device)
            x2 = x2.to(Param.device)
            x3 = x3.to(Param.device)

            if data_type == 'PAIR':
                output1, output2 = model(x1, x2)
                output3 = x3
            else:
                output1, output2, output3 = model(x1, x2, x3)

            loss_value = loss_func.forward(output1, output2, output3).item()

            val_loss = val_loss + ((loss_value - val_loss) / (i + 1))
            val_acc = val_acc + ((get_acc(output1, output2, output3, threshold, data_type) - val_acc) / (i + 1))

    return val_loss, val_acc

def get_val_test_metrics(model, dataset, loss_func):
    model.eval()
    model.zero_grad()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x1, x2 , labels.int() = data
            
            x1 = x1.to(Param.device)
            x2 = x2.to(Param.device)
            labels = labels.to(Param.device)

            optimizer.zero_grad()

            outputs = model(x1, x2)

            loss_value = criterion(outputs, labels).item()

            # get loss and acc train
            vall_loss = vall_loss + ((loss_value - vall_loss) / (i + 1))
            
            y_true = labels.cpu().numpy()
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            acc = accuracy_score(y_true, y_pred)
            val_acc = val_acc + ((acc - val_acc) / (i + 1))

    return val_loss, val_acc