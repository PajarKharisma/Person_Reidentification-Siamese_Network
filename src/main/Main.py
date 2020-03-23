import sys
import os
sys.path.append("../")

import dataPreparation.CreateCsv as create_csv

from config.Path import *

def main():
    # create_csv.contrastive_data(images_path=Path.images, save_path=Path.contrastive_train_csv)
    create_csv.triplet_data(images_path=Path.images, save_path=Path.triplet_train_csv)

if __name__ == "__main__":
    main()