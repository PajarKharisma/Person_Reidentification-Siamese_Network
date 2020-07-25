import torch
import torch.nn.functional as F
from torch.autograd import Variable

import copy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

from src.config.Param import *

def normalize_dist(dist, max_val=-1):
    if max_val == -1:
        max_val = np.max(dist)
    
    norm_dist = (dist - 0) / (max_val - 0)
    norm_dist

def concatenate(dists, thresh, probs):
    result_dist = 0
    result_thresh = 0
    for d, t, p in zip(dists, thresh, probs):
        if d < t and abs(d - t) >= 0.05:
            d = 0
        result_dist += (d * p)
        result_thresh += (t * p)
    return result_dist, result_thresh

def get_distances(x1, x2):
    return F.pairwise_distance(x1, x2, keepdim = True)

def distance_to_class(distances, threshold=0.5, margin=2.0):
    if Param.data_type == 'PAIR':
        y = [0.0 if d <= threshold else 1.0 for d in distances[0]]
    else:
        dist_p = distances[0]
        dist_n = distances[1]
        y = []
        for i in range(len(dist_p)):
            dist = 0.0 if dist_p[i] + margin <= dist_n[i] else 1.0
            y.append(dist)
    return np.array(y)

def get_acc(x1, x2, x3):
    if Param.data_type == 'PAIR':
        y_true = x3.flatten().cpu().numpy()
        distances = [get_distances(x1, x2)]
        y_preds = [distance_to_class(distances=distances, threshold=thresh) for thresh in Param.threshold]
        accs = [accuracy_score(y_true, y_pred) for y_pred in y_preds]
        Param.threshold_list.append(Param.threshold[np.argmax(accs)])
        acc = max(accs)
    else:
        y_true = np.full((len(x1)), 0.0)
        dist_a = get_distances(x1,x2)
        dist_b = get_distances(x1,x3)
        distances = [dist_a, dist_b]
        y_pred = distance_to_class(distances=distances)
        acc = accuracy_score(y_true, y_pred)
    
    return acc

def get_loss(model, dataset, loss_func):
    model.eval()
    model.zero_grad()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x1, x2, x3 = data
            x1 = Variable(x1.to(Param.device))
            x2 = Variable(x2.to(Param.device))
            x3 = Variable(x3.to(Param.device))

            if Param.data_type == 'PAIR':
                output1, output2 = model(x1, x2)
                output3 = x3
            else:
                output1, output2, output3 = model(x1, x2, x3)

            loss_value = loss_func.forward(output1, output2, output3).item()

            val_loss = val_loss + ((loss_value - val_loss) / (i + 1))

    return val_loss

def get_roc_auc(model, dataset):
    model.eval()
    model.zero_grad()

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

                dist = get_distances(output1, output2)
                y_true = torch.cat((y_true, output3))
                y_scores = torch.cat((y_scores, dist))

            else:
                output1, output2, output3 = model(x1, x2, x3)
    
    y_true = y_true.flatten().detach().cpu().numpy()
    y_scores = y_scores.flatten().detach().cpu().numpy()
    y_scores = normalize_dist(y_scores)
    print(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    threshold = roc_auc_score(y_true, y_scores)
    y_pred = distance_to_class([y_scores], threshold)
    
    acc = accuracy_score(y_true, y_pred)

    print('y_true : {}'.format(y_true))
    print('y_score : {}'.format(y_scores))
    print('acc : {}'.format(acc))
    return threshold, acc, (fpr, tpr)