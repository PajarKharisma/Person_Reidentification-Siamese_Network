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
from torchsummary import summary

import copy
import pandas as pd
import matplotlib.pyplot as plt

import src.dataPreparation.CreateCsv as create_csv
import src.dataPreparation.CreatePartial as create_partial

import src.neuralNetworksArch.BasicSiamese as bSiamese
import src.neuralNetworksArch.OneShotArch as osArch
import src.utils.Visual as vis
import src.utils.DatasetLoader as dsetLoader
import src.utils.LossFunction as lossFunc
import src.utils.Metrics as metrics

from src.config.Path import *
from src.config.Param import *

SAVE_PLOT_PATH = root_dir+'/log/plot/'
DATA_SPLIT = 0.9
THRESHOLD = 0.5

def partial_process():
    # create_csv.contrastive_data(images_path=Path.images, save_path=Path.contrastive_train_csv)
    # create_csv.triplet_data(images_path=Path.images, save_path=Path.triplet_train_csv)
    create_partial.create_data(images_path=Path.images, head_path=Path.head_images, body_path=Path.body_images, leg_path=Path.leg_images)

def contrastive_load_process():
    trans = transforms.Compose([transforms.ToTensor()])
    contrastive_dataset = dsetLoader.ContrastiveDataset(csv_path=Path.contrastive_train_csv, images_path=Path.images, transform=trans)

    train_length = int(len(contrastive_dataset) * DATA_SPLIT)
    val_length = len(contrastive_dataset) - train_length

    train_set, val_set = torch.utils.data.random_split(contrastive_dataset, [train_length, val_length])

    train_dataloader = DataLoader(train_set, batch_size=Param.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=val_length, shuffle=True)

    return train_dataloader, val_dataloader

def triplet_load_process():
    trans = transforms.Compose([transforms.ToTensor()])
    triplet_dataset = dsetLoader.TripletDataset(csv_path=Path.triplet_train_csv, images_path=Path.images, transform=trans)
    
    train_length = int(len(contrastive_dataset) * DATA_SPLIT)
    val_length = len(contrastive_dataset) - train_length

    train_dataloader = DataLoader(train_set, batch_size=Param.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=val_length, shuffle=True)

    return train_dataloader, val_dataloader

def training(model, loss_function, dataset, data_type):
    criterion = loss_function
    train_dataloader, val_dataloader = dataset
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    history_loss = {
        'epoch' : [],
        'train' : [],
        'val' : []
    }

    history_acc = {
        'epoch' : [],
        'train' : [],
        'val' : []
    }

    best_loss = 100
    best_model = None
    
    for epoch in range(0, Param.train_number_epochs):
        train_loss = 0
        train_acc = 0
        net.train()
        iteration = 1
        for i, data in enumerate(train_dataloader):
            x1, x2 , x3 = data
            
            x1 = x1.to(Param.device)
            x2 = x2.to(Param.device)
            x3 = x3.to(Param.device)

            optimizer.zero_grad()
            if data_type == 'PAIR':
                output1, output2 = model(x1, x2)
                output3 = x3
            else:
                output1, output2, output3 = model(x1, x2, x3)

            loss_value = criterion(output1, output2, output3)
            loss_value.backward()

            optimizer.step()

            # get loss and acc train
            train_loss = train_loss + ((loss_value.item() - train_loss) / iteration)
            train_acc = train_acc + ((metrics.get_acc(output1, output2, label, THRESHOLD, data_type) - train_acc) / iteration)
            iteration += 1

        if train_loss < best_loss:
            best_loss = train_loss
            best_model = copy.deepcopy(net)
        
        val_loss = metrics.get_val_loss(model, criterion, val_dataloader)
        x1, x2, x3 = metrics.validate(model, val_dataloader, data_type)
        val_acc = metrics.get_acc(x1, x2, x3, THRESHOLD)

        print('Epoch Number : {}'.format(epoch + 1))
        print('-'*40)
        print('Train loss : {}'.format(train_loss))
        print('Validation loss : {}'.format(val_loss))
        print('Train acc : {}'.format(train_acc))
        print('Validation acc : {}'.format(val_acc))

        history_acc['epoch'].append(epoch+1)
        history_acc['train'].append(train_acc)
        history_acc['val'].append(val_acc)

        history_loss['epoch'].append(epoch+1)
        history_loss['train'].append(train_loss)
        history_loss['val'].append(val_loss)

        print('='*40, end='\n\n')

    elapsed_time = time.time() - start_time
    print(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))

    torch.save(best_model.state_dict(), Path.model)
    vis.show_plot(
        history=history_acc,
        title='Akurasi Train dan Validasi',
        xlabel='Epoch',
        ylabel='Akurasi',
        legend_loc='upper left',
        path=SAVE_PLOT_PATH+'Model Akurasi.png',
        should_show=False,
        should_save=True
    )
    
    vis.show_plot(
        history=history_loss,
        title='Loss Train dan Validasi',
        xlabel='Epoch',
        ylabel='Loss',
        legend_loc='upper right',
        path=SAVE_PLOT_PATH+'Model loss.png',
        should_show=False,
        should_save=True
    )
    # vis.show_plot(counter,loss_history)

def triplet_train():
    start_time = time.time()
    print('Process...')

    model = bSiamese.BasicSiameseNetwork()
    model.to(Param.device)

    criterion = lossFunc.TripletLoss()
    dataset = contrastive_load_process()

    training(
        model=model,
        loss_function=criterion,
        dataset=dataset,
        data_type='PAIR'
    )

    elapsed_time = time.time() - start_time
    print(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))

    torch.save(best_model.state_dict(), Path.model)
    vis.imsave(counter, loss_history, path=SAVE_PLOT_PATH, xlabel='Epoch', ylabel='loss')

def contrastive_train():
    start_time = time.time()
    print('Process...')

    net = bSiamese.BasicSiameseNetwork()
    net.to(Param.device)

    criterion = lossFunc.ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    train_dataloader, val_dataloader = contrastive_load_process()

    history_loss = {
        'epoch' : [],
        'train' : [],
        'val' : []
    }

    history_acc = {
        'epoch' : [],
        'train' : [],
        'val' : []
    }

    best_loss = 100
    best_model = None
    
    for epoch in range(0, Param.train_number_epochs):
        train_loss = 0
        train_acc = 0
        net.train()
        iteration = 1
        for i, data in enumerate(train_dataloader):
            img0, img1 , label = data
            
            img0 = img0.to(Param.device)
            img1 = img1.to(Param.device)
            label = label.to(Param.device)

            optimizer.zero_grad()
            output1, output2 = net(img0,img1)

            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            # get loss and acc train
            train_loss = train_loss + ((loss_contrastive.item() - train_loss) / iteration)
            train_acc = train_acc + ((metrics.get_acc(output1, output2, label, THRESHOLD) - train_acc) / iteration)
            iteration += 1

        if train_loss < best_loss:
            best_loss = train_loss
            best_model = copy.deepcopy(net)
        
        val_loss = metrics.get_val_loss(net, criterion, val_dataloader)
        x1, x2, label = metrics.contrastive_validate(net, val_dataloader)
        val_acc = metrics.get_acc(x1, x2, label, THRESHOLD)

        print('Epoch Number : {}'.format(epoch + 1))
        print('-'*40)
        print('Train loss : {}'.format(train_loss))
        print('Validation loss : {}'.format(val_loss))
        print('Train acc : {}'.format(train_acc))
        print('Validation acc : {}'.format(val_acc))

        history_acc['epoch'].append(epoch+1)
        history_acc['train'].append(train_acc)
        history_acc['val'].append(val_acc)

        history_loss['epoch'].append(epoch+1)
        history_loss['train'].append(train_loss)
        history_loss['val'].append(val_loss)

        print('='*40, end='\n\n')

    elapsed_time = time.time() - start_time
    print(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))

    torch.save(best_model.state_dict(), Path.model)
    vis.show_plot(
        history=history_acc,
        title='Akurasi Train dan Validasi',
        xlabel='Epoch',
        ylabel='Akurasi',
        legend_loc='upper left',
        path=SAVE_PLOT_PATH+'Model Akurasi.png',
        should_show=False,
        should_save=True
    )
    
    vis.show_plot(
        history=history_loss,
        title='Loss Train dan Validasi',
        xlabel='Epoch',
        ylabel='Loss',
        legend_loc='upper right',
        path=SAVE_PLOT_PATH+'Model loss.png',
        should_show=False,
        should_save=True
    )
    # vis.show_plot(counter,loss_history)

if __name__ == "__main__":
    triplet_train()