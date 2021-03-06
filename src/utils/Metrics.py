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
    
    return (dist - 0) / (max_val - 0)

def concatenate(dists, thresh, probs):
    result_dist = 0
    result_thresh = 0
    for d, t, p in zip(dists, thresh, probs):
        # if d < t and abs(d - t) >= 0.05:
        #     d = 0
        result_dist += (d * p)
        result_thresh += (t * p)
    return result_dist, result_thresh

def get_distances(x1, x2):
    return F.pairwise_distance(x1, x2, keepdim = True)

def distance_to_class(distances, threshold=0.5):
    y = [0.0 if d <= threshold else 1.0 for d in distances]
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
                d1 = get_distances(output1, output2)
                d2 = get_distances(output1, output3)

                y_true = torch.cat((y_true, torch.zeros_like(d1)))
                y_true = torch.cat((y_true, torch.ones_like(d2)))

                y_scores = torch.cat((y_scores, d1))
                y_scores = torch.cat((y_scores, d2))
    
    y_true = y_true.flatten().detach().cpu().numpy()
    y_scores = y_scores.flatten().detach().cpu().numpy()
    max_dist = np.max(y_scores)
    y_scores = normalize_dist(y_scores, max_dist)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    y_pred = distance_to_class(y_scores, best_thresh)
    
    acc = accuracy_score(y_true, y_pred)

    return {
        'best_thresh' : best_thresh,
        'acc' : acc,
        'fpr' : fpr,
        'tpr' : tpr,
        'ix' : ix,
        'max_dist' : max_dist
    }