import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torchvision.utils
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary # type: ignore

import copy
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score # type: ignore

import src.dataPreparation.CreateCsv as create_csv
import src.dataPreparation.CreatePartial as create_partial
import src.dataPreparation.AugmentationData as aug_data
import src.dataPreparation.CreateDataTest as create_datatest

import src.neuralNetworksArch.BasicSiamese as bSiamese
import src.neuralNetworksArch.OneShotArch as osArch
import src.neuralNetworksArch.AdaptiveSpatialFeature as asf
import src.neuralNetworksArch.BstCnn as bst
import src.neuralNetworksArch.BstCnnFull as bst_full
import src.neuralNetworksArch.McbCnn as mcb
import src.neuralNetworksArch.VggArch as vgg

import src.utils.Visual as vis
import src.utils.DatasetLoader as dsetLoader
import src.utils.LossFunction as lossFunc
import src.utils.Metrics as metrics
import src.utils.Checkpoint as ckp

from src.config.Path import *
from src.config.Param import *

def partial_process():
    create_csv.contrastive_data(images_path=Path.test_images, save_path=Path.test_csv)
    # create_csv.triplet_data(images_path=Path.images, save_path=Path.triplet_train_csv)
    # create_partial.create_data(
    #     images_path=Path.images,
    #     save_path=(
    #         Path.part_1_images,
    #         Path.part_2_images,
    #         Path.part_3_images
    #     )
    # )

def create_datatest_process():
    # create_datatest.create_csv(src_path=Path.contrastive_train_csv, dst_path=Path.test_csv)
    # create_datatest.get_images(csv_path=Path.test_csv, img_src_path=Path.train_images, img_dst_path=Path.test_images)

    occl_data = [
        {
            'save_path' : Path.test_images_20,
            'occlusion' : 0.2
        },
        {
            'save_path' : Path.test_images_40,
            'occlusion' : 0.4
        },
        {
            'save_path' : Path.test_images_60,
            'occlusion' : 0.6
        },
        {
            'save_path' : Path.test_images_80,
            'occlusion' : 0.8
        },
    ]

    for data in occl_data:
        create_datatest.create_ocl_data(
            img_src_path=Path.test_images,
            img_dst_path=data['save_path'],
            occlusion=data['occlusion'],
            occl_pos=-1
        )

def contrastive_load_process(split_data = True):
    trans = transforms.Compose([transforms.ToTensor()])
    contrastive_dataset = dsetLoader.ContrastiveDataset(
        csv_path=Path.contrastive_train_csv,
        images_path=Path.train_images,
        transform=trans,
        resize=Param.input_size,
        count=20000
    )

    if split_data:
        train_length = int(len(contrastive_dataset) * Param.data_split)
        val_length = len(contrastive_dataset) - train_length

        train_set, val_set = torch.utils.data.random_split(contrastive_dataset, [train_length, val_length])

        train_dataloader = DataLoader(train_set, batch_size=Param.train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=Param.train_batch_size * 2, shuffle=True)

        return train_dataloader, val_dataloader
    else:
        return DataLoader(contrastive_dataset, batch_size=Param.train_batch_size, shuffle=True)

def triplet_load_process(split_data = True):
    trans = transforms.Compose([transforms.ToTensor()])
    triplet_dataset = dsetLoader.TripletDataset(
        csv_path=Path.triplet_train_csv,
        images_path=Path.train_images,
        transform=trans,
        resize=Param.input_size
    )

    if split_data:
        train_length = int(len(triplet_dataset) * Param.data_split)
        val_length = len(triplet_dataset) - train_length

        train_set, val_set = torch.utils.data.random_split(triplet_dataset, [train_length, val_length])

        train_dataloader = DataLoader(train_set, batch_size=Param.train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=Param.train_batch_size * 2, shuffle=True)

        return train_dataloader, val_dataloader
    else:
        return DataLoader(triplet_dataset, batch_size=Param.train_batch_size, shuffle=True)

def training(model, loss_function, dataset, optimizer, loss, epoch_number=0):
    criterion = loss_function
    train_dataloader, val_dataloader = dataset
    optimizer = optimizer

    history_loss = {
        'epoch' : [],
        'train' : [],
        'val' : []
    }

    best_loss = loss
    val_model = None
    best_model = None

    if Param.pretrained == True:
        best_model = copy.deepcopy(model)

    for epoch in range(0, Param.train_number_epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for i, data in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            x1, x2 , x3 = data
            
            x1 = Variable(x1.to(Param.device))
            x2 = Variable(x2.to(Param.device))
            x3 = Variable(x3.to(Param.device))

            optimizer.zero_grad()
            if Param.data_type == 'PAIR':
                output1, output2 = model(x1, x2)
                output3 = x3
            else:
                output1, output2, output3 = model(x1, x2, x3)

            loss_value = criterion(output1, output2, output3)
            loss_value.backward()

            optimizer.step()

            train_loss = train_loss + ((loss_value.item() - train_loss) / (i + 1))

        if train_loss < best_loss:
            best_loss = train_loss
            best_model = copy.deepcopy(model)
        
        val_model = copy.deepcopy(model)
        val_loss = metrics.get_loss(val_model, val_dataloader, criterion)

        output_str = ''
        output_str += 'Epoch Number : {}'.format(epoch + 1) + '\n'
        output_str += '-'*40 + '\n'
        output_str += 'Train loss : {}'.format(train_loss) + '\n'
        output_str += 'Validation loss : {}'.format(val_loss) + '\n'

        sys.stdout.write(output_str)
        sys.stdout.flush()

        history_loss['epoch'].append(epoch+1)
        history_loss['train'].append(train_loss)
        history_loss['val'].append(val_loss)

        print('='*40, end='\n\n')

    # test_dataset = contrastive_load_process(split_data = False)
    roc_data = metrics.get_roc_auc(best_model, val_dataloader)

    sys.stdout.write('Akurasi data validasi : {}\n'.format(roc_data['acc']))
    sys.stdout.flush()

    vis.show_plot(
        type='val',
        epoch=history_loss['epoch'],
        train_data=history_loss['train'],
        val_data=history_loss['val'],
        title='Loss Train dan Validasi',
        xlabel='Epoch',
        ylabel='Loss',
        legend=['Train', 'Val'],
        path=Path.save_plot+'Model loss.png',
        should_show=False,
        should_save=True
    )

    vis.show_plot(
        type='roc',
        fpr=roc_data['fpr'],
        tpr=roc_data['tpr'],
        title='ROC Curve',
        best_thresh=roc_data['best_thresh'],
        ix=roc_data['ix'],
        path=Path.save_plot+'ROC.png',
        should_show=False,
        should_save=True
    )

    ckp.save_checkpoint(
        best_threshold=roc_data['best_thresh'],
        save_dir=Path.save_model,
        model=best_model,
        optimizer=optimizer,
        epoch=Param.train_number_epochs + epoch_number,
        loss=best_loss,
        max_dist=roc_data['max_dist']
    )
    # torch.save(best_model.state_dict(), Path.model)

def renew_model():
    details = [
        # PARTIAL_1
        {
            'load_model' : Path.load_model + '/PARTIAL_1 #1.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_1 #1.pth',
            'image_path' : Path.path + '/images/partial_1/part_1/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_1 #1.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_1 #1.out'
        },
        {
            'load_model' : Path.load_model + '/PARTIAL_1 #2.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_1 #2.pth',
            'image_path' : Path.path + '/images/partial_1/part_2/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_1 #2.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_1 #2.out'
        },
        {
            'load_model' : Path.load_model + '/PARTIAL_1 #3.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_1 #3.pth',
            'image_path' : Path.path + '/images/partial_1/part_3/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_1 #3.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_1 #3.out'
        },
        # PARTIAL_2
        {
            'load_model' : Path.load_model + '/PARTIAL_2 #1.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_2 #1.pth',
            'image_path' : Path.path + '/images/partial_2/part_1/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_2 #1.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_2 #1.out'
        },
        {
            'load_model' : Path.load_model + '/PARTIAL_2 #2.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_2 #2.pth',
            'image_path' : Path.path + '/images/partial_2/part_2/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_2 #2.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_2 #2.out'
        },
        {
            'load_model' : Path.load_model + '/PARTIAL_2 #3.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_2 #3.pth',
            'image_path' : Path.path + '/images/partial_2/part_3/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_2 #3.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_2 #3.out'
        },
        # PARTIAL_3
        {
            'load_model' : Path.load_model + '/PARTIAL_3 #1.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_3 #1.pth',
            'image_path' : Path.path + '/images/partial_3/part_1/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_3 #1.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_3 #1.out'
        },
        {
            'load_model' : Path.load_model + '/PARTIAL_3 #2.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_3 #2.pth',
            'image_path' : Path.path + '/images/partial_3/part_2/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_3 #2.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_3 #2.out'
        },
        {
            'load_model' : Path.load_model + '/PARTIAL_3 #3.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_3 #3.pth',
            'image_path' : Path.path + '/images/partial_3/part_3/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_3 #3.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_3 #3.out'
        },
        {
            'load_model' : Path.load_model + '/PARTIAL_3 #4.pth',
            'save_model' : Path.root_dir + '/models/PARTIAL_3 #4.pth',
            'image_path' : Path.path + '/images/partial_3/part_4/',
            'plot_name' : Path.root_dir + '/log/plot/ROC PARTIAL_3 #4.png',
            'out_log' : Path.root_dir + '/log/result/data roc - PARTIAL_3 #4.out'
        }
    ]

    for detail in details:
        model  = bst.BstCnn()
        checkpoint = ckp.load_checkpoint(load_dir=detail['load_model'])
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(Param.device)
        model.eval()

        contrastive_dataset = dsetLoader.ContrastiveDataset(
            csv_path=Path.contrastive_train_csv,
            images_path=detail['image_path'],
            transform=transforms.Compose([transforms.ToTensor()]),
            resize=Param.input_size
        )
        dataset = DataLoader(contrastive_dataset, batch_size=Param.train_batch_size, shuffle=True)

        roc_data = metrics.get_roc_auc(model, dataset)

        output = ''
        output += checkpoint['desc'] + '\n\n'
        output += 'Threshold : {}\n'.format(roc_data['best_thresh'])
        output += 'Max distance : {}\n'.format(roc_data['max_dist'])
        output += 'Akurasi : {}\n'.format(roc_data['acc'])

        with open(detail['out_log'], "w+") as text_file:
            text_file.write(output)

        vis.show_plot(
            type='roc',
            fpr=roc_data['fpr'],
            tpr=roc_data['tpr'],
            title='ROC Curve',
            best_thresh=roc_data['best_thresh'],
            ix=roc_data['ix'],
            path=detail['plot_name'],
            should_show=False,
            should_save=True
        )

        checkpoint = {
            'desc' : checkpoint['desc'],
            'threshold' : roc_data['best_thresh'],
            'epoch': checkpoint['epoch'],
            'loss' : checkpoint['loss'],
            'state_dict': checkpoint['state_dict'],
            'optimizer': checkpoint['optimizer'],
            'max_dist' : roc_data['max_dist']
        }
        torch.save(checkpoint, detail['save_model'])

def contrastive_train():
    model = bst_full.BstCnnFull()
    model = model.to(Param.device)

    optimizer = optim.Adam(model.parameters())
    epoch = 0
    loss = sys.float_info.max

    if(Param.pretrained == True):
        checkpoint  = ckp.load_checkpoint(load_dir=Path.load_model)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        Param.threshold = checkpoint['threshold']

    criterion = lossFunc.ContrastiveLoss()
    sys.stdout.write('# READING DATASET\n')
    sys.stdout.flush()

    dataset = contrastive_load_process()

    sys.stdout.write('# FINISHED READING DATASET AND START TRAINING\n\n')
    sys.stdout.flush()

    training(
        model=model,
        loss_function=criterion,
        dataset=dataset,
        optimizer=optimizer,
        loss=loss,
        epoch_number=epoch
    )

def triplet_train():
    model = bst_full.BstCnnFull()
    model = model.to(Param.device)

    optimizer = optim.Adam(model.parameters())
    epoch = 0
    loss = sys.float_info.max

    if(Param.pretrained == True):
        checkpoint  = ckp.load_checkpoint(load_dir=Path.load_model)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        Param.threshold = checkpoint['threshold']

    criterion = lossFunc.TripletLoss()
    sys.stdout.write('# READING DATASET\n')

    dataset = triplet_load_process()

    sys.stdout.write('# FINISH READING DATASET AND START TRAINING\n\n')
    sys.stdout.flush()

    training(
        model=model,
        loss_function=criterion,
        dataset=dataset,
        optimizer=optimizer,
        loss=loss,
        epoch_number=epoch
    )

if __name__ == "__main__":
    start_time = time.time()
    sys.stdout.write('Process using '+str(Param.device)+'\n')
    sys.stdout.write(Param.desc+'\n\n')
    sys.stdout.flush()

    # renew_model()
    contrastive_train()
    # partial_process()
    # create_datatest_process()

    elapsed_time = time.time() - start_time
    print(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))