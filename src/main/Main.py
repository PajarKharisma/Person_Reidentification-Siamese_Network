import sys
import os
import time
sys.path.append("../")

import dataPreparation.CreateCsv as create_csv
import dataPreparation.CreatePartial as create_partial

from config.Path import *

def main():
    start_time = time.time()
    print('Process...')
    # create_csv.contrastive_data(images_path=Path.images, save_path=Path.contrastive_train_csv)
    # create_csv.triplet_data(images_path=Path.images, save_path=Path.triplet_train_csv)
    create_partial.create_data(images_path=Path.images, head_path=Path.head_images, body_path=Path.body_images, leg_path=Path.leg_images)
    
    elapsed_time = time.time() - start_time
    finish_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(finish_time)

if __name__ == "__main__":
    main()