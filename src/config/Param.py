import torch
import sys
import datetime
import numpy as np

class Param():
    #Hyperparameter
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 32
    train_number_epochs = 100
    data_split = 0.8
    input_size = (64,128)
    threshold = np.arange(0.1, 1, 0.05)
    pretrained = False
    data_type = 'PAIR'
    desc = 'PARTIAL_3 #1 ' + str(datetime.datetime.now())

    max_dist = 0
    min_dist = sys.float_info.max
    
    threshold_list = []