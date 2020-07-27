import torch
from scipy import stats

from src.config.Path import *
from src.config.Param import *

def save_checkpoint(**kwargs):
    data = kwargs
    checkpoint = {
        'desc' : Param.desc,
        'threshold' : data['best_threshold'],
        'epoch': data['epoch'],
        'loss' : data['loss'],
        'state_dict': data['model'].state_dict(),
        'optimizer': data['optimizer'].state_dict(),
        'max_dist' : data['max_dist']
    }
    torch.save(checkpoint, data['save_dir'])

def load_checkpoint(load_dir):
    checkpoint = torch.load(load_dir, map_location=Param.device)

    return checkpoint