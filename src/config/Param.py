import torch

class Param():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 64
    train_number_epochs = 10