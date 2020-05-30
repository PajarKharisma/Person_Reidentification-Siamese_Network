import os
from datetime import datetime

root_dir = os.getcwd()

class Path():
    path = root_dir + '/dataset/cuhk02'
    model = root_dir + '/model/model-' + str(datetime.now())

    images = path + '/images/full/'
    head_images = path + '/images/head/'
    body_images = path + '/images/body/'
    leg_images =path + '/images/leg/'

    contrastive_train_csv = path +'/contrastive_train.csv'
    triplet_train_csv = path + '/triplet_train.csv'
    testing_csv = path + '/testing.csv'
