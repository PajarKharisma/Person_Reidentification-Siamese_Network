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
    save_plot = root_dir+'/log/plot/'

    train_images = path + '/images/full/'
    partial_path = path + '/images/partial_3/'
    part_1_images = partial_path + '/part_1/'
    part_2_images = partial_path + '/part_2/'
    part_3_images =partial_path + '/part_3/'
    part_4_images =partial_path + '/part_4/'

    contrastive_train_csv = path +'/contrastive_train.csv'
    triplet_train_csv = path + '/triplet_train.csv'

    test_path = root_dir + '/dataset/testing/same_cam'
    test_csv = test_path + '/testing.csv'
    test_images = test_path + '/images/full/'
    test_images_20 = test_path + '/images/occl_20/'
    test_images_40 = test_path + '/images/occl_40/'
    test_images_60 = test_path + '/images/occl_60/'
    test_images_80 = test_path + '/images/occl_80/'
