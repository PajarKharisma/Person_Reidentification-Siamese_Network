import sys
import os
import time
root_dir = os.getcwd()
sys.path.append(root_dir)

import torch
import torchvision.utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd

import src.dataPreparation.CreateCsv as create_csv
import src.dataPreparation.CreatePartial as create_partial

import src.utils.Visual as vis
from src.utils.DatasetLoader import *

from src.config.Path import *

def main():
    start_time = time.time()
    print('Process...')
    # create_csv.contrastive_data(images_path=Path.images, save_path=Path.contrastive_train_csv)
    # create_csv.triplet_data(images_path=Path.images, save_path=Path.triplet_train_csv)
    # create_partial.create_data(images_path=Path.images, head_path=Path.head_images, body_path=Path.body_images, leg_path=Path.leg_images)

    trans = transforms.Compose([transforms.ToTensor()])
    contrastive_dataset = ContrastiveDataset(csv_path=Path.contrastive_train_csv, images_path=Path.images, transform=trans)
    contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=8, shuffle=True)
    dataiter = iter(contrastive_dataloader)
    
    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)

    print(example_batch[2].numpy())
    vis.imshow(torchvision.utils.make_grid(concatenated))
    
    # vis.imshow(x1)

    elapsed_time = time.time() - start_time
    print(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))

if __name__ == "__main__":
    main()