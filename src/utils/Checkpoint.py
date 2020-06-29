import torch
from src.config.Path import *
from src.config.Param import *

def save_checkpoint(desc, save_dir, model, optimizer, loss, epoch, dist):
    checkpoint = {
        'desc' : desc,
        'epoch': epoch,
        'loss' : loss,
        'dist' : dist,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)

def load_checkpoint(load_dir, model, optimizer):
    checkpoint = torch.load(load_dir, map_location=Param.device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, checkpoint['epoch'], checkpoint['loss'], checkpoint['dist'], desc