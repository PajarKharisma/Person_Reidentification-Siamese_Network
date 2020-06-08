import torch
import torch.nn.functional as F

import copy
import numpy as np
from sklearn.metrics import accuracy_score

from src.config.Param import *

def distance_to_class(distances, threshold=0.5):
    distances = distances.flatten().detach().cpu().numpy()
    distances_norm = [abs((1 / (1 + d)) - 1) for d in distances]
    y = [0.0 if d <= threshold else 1.0 for d in distances_norm]

    return np.array(y)

def get_distances(x1, x2):
    return F.pairwise_distance(x1, x2, keepdim = True)

def get_acc(x1, x2, x3, threshold=0.5, data_type='PAIR'):
    if data_type == 'PAIR':   
        y_true = x3.flatten().cpu().numpy()
    else:
        y_true = np.full((1, len(x1)), 1.0)
    
    distances = get_distances(x1, x2)
    y_pred = distance_to_class(distances, threshold)
    return accuracy_score(y_true, y_pred)

def get_val_loss(base_model, loss_func, dataset, data_type='PAIR'):
    model = copy.deepcopy(base_model)
    model.eval()
    with torch.no_grad():
        dataiter = iter(dataset)
        x1, x2, x3 = next(dataiter)

        x1 = x1.to(Param.device)
        x2 = x2.to(Param.device)
        x3 = x3.to(Param.device)

        if data_type == 'PAIR':
            output1, output2 = model(x1,x2)
        else:
            output1, output2, x3 = model(x1, x2, x3)

    return loss_func.forward(output1, output2, x3).item()

def validate(base_model, dataset, data_type='PAIR'):
    model = copy.deepcopy(base_model)
    model.eval()
    with torch.no_grad():
        dataiter = iter(dataset)
        x1, x2, x3 = next(dataiter)

        x1 = x1.to(Param.device)
        x2 = x2.to(Param.device)
        x3 = x3.to(Param.device)

        if data_type == 'PAIR':
            output1, output2 = model(x1,x2)
            output3 = x3
        else:
            output1, output2, output3 = model(x1,x2,x3)

        return output1, output2, output3
    