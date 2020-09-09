import torch
import sys
import datetime
import numpy as np

class Param():
    #Hyperparameter
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 32
    train_number_epochs = 50
    data_split = 0.8
    learning_rate = 0.1
    input_size = (64,128)
    pretrained = False
    data_type = 'PAIR'
    desc = 'Train Epoch 50 ' + str(datetime.datetime.now())

    threshold = -1