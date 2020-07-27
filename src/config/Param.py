import torch
import sys
import datetime
import numpy as np

class Param():
    #Hyperparameter
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 32
    train_number_epochs = 100
    data_split = 0.9
    input_size = (64,128)
    pretrained = False
    data_type = 'PAIR'
    desc = 'RENEW ALL MODEL ' + str(datetime.datetime.now())

    threshold = -1