import cv2
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ContrastiveDataset(Dataset):
    def __init__(self, csv_path, images_path, transform=None, should_invert=True):
        self.images_path = images_path
        self.csv_path = csv_path
        self.transform = transform
        self.should_invert = should_invert
        self.df = pd.read_csv(self.csv_path)
    
    def __getitem__(self, index):
        img1 = cv2.imread(self.images_path + self.df['image_1'][index])
        img2 = cv2.imread(self.images_path + self.df['image_2'][index])
        label = int(self.df['label'][index])

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))
    
    def __len__(self):
        return len(self.df)

class TripletDataset(Dataset):
    def __init__(self, csv_path, images_path, transform=None, should_invert=True):
        self.images_path = images_path
        self.csv_path = csv_path
        self.transform = transform
        self.should_invert = should_invert
        self.df = pd.read_csv(self.csv_path)
    
    def __getitem__(self, index):
        anc = cv2.imread(self.images_path + self.df['anchor'][index])
        pos = cv2.imread(self.images_path + self.df['positif'][index])
        neg = cv2.imread(self.images_path + self.df['negatif'][index])

        anc = cv2.cvtColor(anc, cv2.COLOR_BGR2RGB)
        pos = cv2.cvtColor(pos, cv2.COLOR_BGR2RGB)
        neg = cv2.cvtColor(neg, cv2.COLOR_BGR2RGB)

        anc = self.transform(anc)
        pos = self.transform(pos)
        neg = self.transform(neg)
        
        return anc, pos, neg
    
    def __len__(self):
        return len(self.df)