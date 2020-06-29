import torch
import sys

class Param():
    #Hyperparameter
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 32
    train_number_epochs = 10
    data_split = 0.8
    input_size = (64,128)
    threshold = 0.5
    pretrained = True
    data_type = 'PAIR'

    max_dist = 0
    min_dist = sys.float_info.max