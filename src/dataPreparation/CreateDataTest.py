import pandas as pd
import random

def create_csv(path_src, path_dst):
    df_src = pd.read_csv(path_src)
    n_src = len(df_src)
    n_dst = random.sample(range(1, n_src), 200)

    data = []
    for i in n_dst:
        pair_data = {}
        pair_data['image_1'] = df_src['image_1'][i]
        pair_data['image_2'] = df_src['image_2'][i]
        pair_data['label'] = df_src['label'][i]
        data.append(pair_data)
    
    df = pd.DataFrame(data)
    df.to_csv(path_dst, index=False)