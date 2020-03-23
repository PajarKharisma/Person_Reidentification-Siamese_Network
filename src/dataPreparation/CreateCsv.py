import cv2
import pandas as pd
import os

NUM_PAIRS = 30

def data_contrastive(images_path='', save_path=''):
    files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(images_path):
        data_length = len(f)
        for index in range(data_length):
            id, _ = f[index].split('_')
            list_id = [id]
            isSameId = True
            next_index = index + 1

            while isSameId:
                next_id, _ = f[next_index].split('_')
                if next_id == id:
                    list.append(next_id)
                else:
                    isSameId = False

            data = {}
            data['img1'] = id
            data['no'] = _
            files.append(data)
    
    df = pd.DataFrame(files)
    df.to_csv(save_path, index=False)
            