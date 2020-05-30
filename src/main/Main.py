import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

import torch
import torchvision.utils
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt

import src.dataPreparation.CreateCsv as create_csv
import src.dataPreparation.CreatePartial as create_partial

import src.utils.Visual as vis
from src.utils.DatasetLoader import *
from src.utils.LossFunction import *
from src.utils.NeuralNetworks import *

from src.config.Path import *
from src.config.Param import *

def partial_process():
    create_csv.contrastive_data(images_path=Path.images, save_path=Path.contrastive_train_csv)
    create_csv.triplet_data(images_path=Path.images, save_path=Path.triplet_train_csv)
    create_partial.create_data(images_path=Path.images, head_path=Path.head_images, body_path=Path.body_images, leg_path=Path.leg_images)

def contrastive_load_process():
    trans = transforms.Compose([transforms.ToTensor()])
    contrastive_dataset = ContrastiveDataset(csv_path=Path.contrastive_train_csv, images_path=Path.images, transform=trans)
    contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=Param.train_batch_size, shuffle=True)

    return contrastive_dataloader

def triplet_load_process():
    trans = transforms.Compose([transforms.ToTensor()])
    triplet_dataset = TripletDataset(csv_path=Path.triplet_train_csv, images_path=Path.images, transform=trans)
    triplet_dataloader = DataLoader(triplet_dataset, batch_size=1, shuffle=True)
    dataiter = iter(triplet_dataloader)
    
    example_batch = next(dataiter)
    
    concatenated = torch.cat((example_batch[0],example_batch[1], example_batch[2]),0)
    vis.imshow(torchvision.utils.make_grid(concatenated))

def main():
    start_time = time.time()
    print('Process...')

    net = BasicSiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    train_dataloader = contrastive_load_process()

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, Param.train_number_epochs):
        print('Epoch Number', epoch)
        for i, data in enumerate(train_dataloader, 0):
            img0, img1 , label = data
            optimizer.zero_grad()
            output1, output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            val = criterion.forward(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if True or i % 10 == 0:
                print(i)
                print("Iteration {}\n Current loss {}\n".format(i,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        print('='*40)

    elapsed_time = time.time() - start_time
    print(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))

    vis.show_plot(counter,loss_history)
    torch.save(net, Path.model)

if __name__ == "__main__":
    main()