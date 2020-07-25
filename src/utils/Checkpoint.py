import torch
from scipy import stats

from src.config.Path import *
from src.config.Param import *

def save_checkpoint(desc, save_dir, model, optimizer, loss, epoch, dist):
    checkpoint = {
        'desc' : desc,
        'threshold' : Param.threshold,
        'epoch': epoch,
        'loss' : loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)

def load_checkpoint(load_dir):
    checkpoint = torch.load(load_dir, map_location=Param.device)

    return checkpoint