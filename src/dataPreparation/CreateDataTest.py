import pandas as pd
import random
import cv2
import os

import src.dataPreparation.AugmentationData as aug_data

NUM_DATA = 5

def create_csv(src_path, dst_path):
    df_src = pd.read_csv(src_path)

    data_sim = []
    data_dis = []
    for index, data in df_src.iterrows():
        if int(data['label']) == 0:
            data_sim.append(index)
        else:
            data_dis.append(index)


    data_sim = random.sample(data_sim, NUM_DATA)
    data_dis = random.sample(data_dis, NUM_DATA)

    data_dst = [*data_sim, *data_dis]
    data_test = []
    for i in data_dst:
        pair_data = {}
        pair_data['image_1'] = df_src['image_1'][i]
        pair_data['image_2'] = df_src['image_2'][i]
        pair_data['label'] = df_src['label'][i]
        data_test.append(pair_data)
    
    df = pd.DataFrame(data_test)
    df.to_csv(dst_path, index=False)

def get_images(csv_path, img_src_path, img_dst_path):
    df = pd.read_csv(csv_path)
    img_names = [*df['image_1'], *df['image_2']]
    img_names = set(img_names)

    for img_name in img_names:
        img = cv2.imread(img_src_path + img_name)
        cv2.imwrite(img_dst_path + img_name, img)

def create_ocl_data(img_src_path, img_dst_path, occlusion, occl_pos=aug_data.BOTTOM, color=(255,255,255)):
    imgs = os.listdir(img_src_path)
    for img_name in imgs:
        img = cv2.imread(img_src_path + img_name)
        img = aug_data.create_aug_data(
            img=img,
            occlusion=occlusion,
            pos=occl_pos,
            color=color
        )

        cv2.imwrite(img_dst_path + img_name, img)

