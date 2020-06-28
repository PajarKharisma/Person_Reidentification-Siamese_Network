import torch
from src.config.Path import *
from src.config.Param import *

def save_checkpoint(save_dir, model, optimizer, epoch):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)

def load_checkpoint(load_dir, model, optimizer):
    checkpoint = torch.load(load_dir)
    model.load_state_dict(checkpoint['state_dict']).to(Param.device)
    optimizer.load_state_dict(checkpoint['optimizer']).to(Param.device)

    return model, optimizer, checkpoint['epoch']