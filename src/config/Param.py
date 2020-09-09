import torch
import sys
import datetime
import numpy as np

class Param():
    #Hyperparameter
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 64
    train_number_epochs = 20
    data_split = 0.8
    learning_rate = 0.1
    input_size = (64,128)
    pretrained = False
    data_type = 'PAIR'
    desc = 'Train Batch Size 64 ' + str(datetime.datetime.now())

    threshold = -1