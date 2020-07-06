import os
import uuid
import platform

class Path():
    # root_dir = os.getcwd() if platform.system() == 'Windows' else '/work/dike003'
    root_dir = os.getcwd()
    path = root_dir + '/dataset/cuhk02'
    log_dir = root_dir + '/log/'
    save_model = root_dir + '/models/model-' + str(uuid.uuid4().hex) + '.pth'
    load_model = root_dir + '/models/pretrained-model.pth'
    save_plot = root_dir+'/log/plot/'

    images = path + '/images/full/'
    partial_path = path + '/images/partial_2/'
    part_1_images = partial_path + '/part_1/'
    part_2_images = partial_path + '/part_2/'
    part_3_images =partial_path + '/part_3/'
    part_4_images =partial_path + '/part_4/'

    contrastive_train_csv = path +'/contrastive_train.csv'
    triplet_train_csv = path + '/triplet_train.csv'
    testing_csv = path + '/testing.csv'
