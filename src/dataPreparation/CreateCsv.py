import cv2
import pandas as pd
import os

from itertools import combinations
from random import randrange

NUM_PAIRS = 10

def get_files(images_path='', save_path=''):
    list_files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(images_path):
        data_length = len(f)
        index = 0
        while index < data_length:
            id, _ = f[index].split('_')
            list_id = []
            isSameId = True
            next_index = index

            while isSameId and next_index < data_length:
                next_id, _ = f[next_index].split('_')
                if next_id == id:
                    list_id.append(f[next_index])
                    next_index += 1
                else:
                    isSameId = False
            
            list_files.append(list_id)
            index = next_index
    return list_files

def data_contrastive(images_path='', save_path=''):
    data = []
    list_files = get_files(images_path=images_path, save_path=save_path)

    for index, files in enumerate(list_files):
        comb = combinations(files, 2)
        list_comb = list(comb)

        for pairs in list_comb:
            pair_data = {}
            img1, img2 = pairs

            pair_data['image_1'] = img1
            pair_data['image_2'] = img2
            pair_data['label'] = 0

            data.append(pair_data)

        count = 0
        while count < len(list_comb):
            for file in files:
                pair_id = randrange(0, len(list_files))
                while pair_id == index:
                    pair_id = randrange(0, len(list_files))
                
                pair_file = randrange(0, len(list_files[pair_id]))
                pair_data = {}
                pair_data['image_1'] = file
                pair_data['image_2'] = list_files[pair_id][pair_file]
                pair_data['label'] = 1
                data.append(pair_data)
                count += 1

        
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

def triplet_data(images_path='', save_path=''):
    data = []
    list_files = get_files(images_path=images_path, save_path=save_path)
            