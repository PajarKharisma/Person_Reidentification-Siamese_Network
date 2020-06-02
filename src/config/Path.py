import os
import uuid
import platform

class Path():
    root_dir = os.getcwd() if platform.system() == 'Windows' else '/work/dike003'
    path = root_dir + '/dataset/cuhk02'
    log_dir = root_dir + '/log/'
    model = root_dir + '/models/model-' + str(uuid.uuid4().hex) + '.pth'

    images = path + '/images/full/'
    head_images = path + '/images/head/'
    body_images = path + '/images/body/'
    leg_images =path + '/images/leg/'

    contrastive_train_csv = path +'/contrastive_train.csv'
    triplet_train_csv = path + '/triplet_train.csv'
    testing_csv = path + '/testing.csv'
