import sys
import os
import time
sys.path.append("../")

import pandas as pd

import dataPreparation.CreateCsv as create_csv
import dataPreparation.CreatePartial as create_partial

from config.Path import *

def main():
    start_time = time.time()
    print('Process...')
    create_csv.contrastive_data(images_path=Path.images, save_path=Path.contrastive_train_csv)
    # create_csv.triplet_data(images_path=Path.images, save_path=Path.triplet_train_csv)
    # create_partial.create_data(images_path=Path.images, head_path=Path.head_images, body_path=Path.body_images, leg_path=Path.leg_images)
    
    # df = pd.read_csv(Path.contrastive_train_csv)
    # print(df['image_1'][47])
    # print(df['image_2'][47])
    # print(df['label'][47])

    elapsed_time = time.time() - start_time
    print(time.strftime("Finish in %H:%M:%S", time.gmtime(elapsed_time)))

if __name__ == "__main__":
    main()