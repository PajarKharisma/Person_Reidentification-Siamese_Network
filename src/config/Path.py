import os
import uuid
import platform

class Path():
    # root_dir = os.getcwd() if platform.system() == 'Windows' else '/work/dike003'
    root_dir = os.getcwd()
    path = root_dir + '/dataset/cuhk03'
    log_dir = root_dir + '/log/'
    save_model = root_dir + '/models/model-' + str(uuid.uuid4().hex) + '.pth'
    load_model = root_dir + '/models/pretrained-model.pth'
    load_model = root_dir + '/models/model-17572c2e5ab943da935120c7069b3a5e.pth'
    save_plot = root_dir+'/log/plot/'

    images = path + '/images/full/'
    head_images = path + '/images/head/'
    body_images = path + '/images/body/'
    leg_images =path + '/images/leg/'

    contrastive_train_csv = path +'/contrastive_train.csv'
    triplet_train_csv = path + '/triplet_train.csv'
    testing_csv = path + '/testing.csv'
