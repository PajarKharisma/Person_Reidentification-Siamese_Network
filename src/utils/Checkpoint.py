import torch

def save_checkpoint(save_dir, model, optimizer, epoch):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)

def load_checkpoint(load_dir, model, optimizer):
    checkpoint = torch.load(load_dir)
    model.load_state_dict(checkpoint['state_dict'], map_location=Param.device)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, checkpoint['epoch']