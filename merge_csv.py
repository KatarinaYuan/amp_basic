import glob
import os 
import pandas as pd
import numpy as np 
import ipdb


QUERY_DIR = './send_output'
df_all = []
for data_path in glob.glob(QUERY_DIR + '/*.csv'):
    df = pd.read_csv(data_path, index_col=0)
    df_all.append(df)
df_all = pd.concat(df_all)
print("before: ", df_all.shape)
df_all = df_all.drop_duplicates()
print("after: ", df_all.shape)
KEEP_SIZE = 75000
idx = np.random.choice(len(df_all), KEEP_SIZE, replace=False)
df_all = df_all.iloc[idx]
df_all['label'] = [0] * KEEP_SIZE
df_all['text'] = df_all['text'].map(lambda x: "▁".join([_ for _ in x]))
df_all.to_csv(os.path.join(QUERY_DIR, 'all.csv'), index=False, header=True, columns=['label', 'text'])