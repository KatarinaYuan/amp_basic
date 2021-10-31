import pandas as pd
import ipdb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datafile', type=str, required=True)
args = parser.parse_args()
datafile = args.datafile

df = pd.read_csv(datafile)
seq_list = []
for x in df['text']:
    for l in range(21, len(x)+1, 2):
        seq_list.append(x[:l])
new_df = pd.DataFrame()
new_df['label'] = [0] * len(seq_list)
new_df['text'] = seq_list
new_df.to_csv(args.datafile[:-4]+"_variadic_21.csv", index=False)