"""
https://buomsoo-kim.github.io/attention/2020/04/22/Attention-mechanism-20.md/
"""

import copy
from tqdm import tqdm 

import numpy as np 
import pandas as pd 

import torch 

#from formatting.amps_util import preprocess_str2ids


# =============  DO NOT CHANGE ================

IS_CONDITIONAL = 0

MAX_LEN = 60#60 # MAXLEN+1 inlcuding eos
MIN_LEN = 12


PADDING_LEN = MAX_LEN + 2 + IS_CONDITIONAL # <bos>, <cls>, <eos>

VOCAB_STOI = {"<bos>":0, "<cls>":1, "<eos>":2, "<pad>":3, "C":4,"D":5,"E":6,"F":7,"G":8,
            "H":9,"I":10,"K":11,"L":12,"M":13,"N":14,"P":15,"Q":16,"R":17,"S":18,"T":19,
            "V":20,"W":21,"Y":22,"B":23,"U":24,"X":25,"Z":26,"J":27,"O":28, "A":29}
# texar-pytorch use 0 as mask value
VOCAB_ITOS = {v:k for k,v in VOCAB_STOI.items()}
VOCAB_SIZE = len(VOCAB_STOI)

PAD_TOKEN_ID = VOCAB_STOI["<pad>"]
BOS_TOKEN_ID = VOCAB_STOI["<bos>"]
EOS_TOKEN_ID = VOCAB_STOI["<eos>"]
CLS_TOKEN_ID = VOCAB_STOI["<cls>"]

BOS_TOKEN_POSINSEQ = 0
CLS_TOKEN_POSINSEQ = 1

TRAIN_DATA_PATH = '/home2/yxy/project/clamp_ALL/baseline/data/unbalanced_dataset_train.csv'
VAL_DATA_PATH = '/home2/yxy/project/clamp_ALL/baseline/data/unbalanced_dataset_valid.csv'
TEST_DATA_PATH = '/home2/yxy/project/clamp_ALL/baseline/data/unbalanced_dataset_test.csv'
TEST_WITTEN = '/home2/yxy/project/clamp_ALL/baseline/data/witten_dataset_minus_unbalanced_dataset.csv'
#TEST_DATA_PATH = '/home2/swp/bio/oracle/data/tmp.csv'  # a small data for debug
QUERY_DATA_PATH_1 = '/home2/yxy/project/clamp_ALL/baseline/gen_phase1_1w.csv'
QUERY_DATA_PATH_2 = '/home2/yxy/project/clamp_ALL/baseline/gen_phase1_10w.csv'
QUERY_DIR1 = '/home2/yxy/project/clamp_ALL/baseline/result/gen/'
QUERY_DIR2 = '/home2/yxy/project/clamp_ALL/baseline/tmp_raw/'

SAVE_PATH = '/home2/swp/bio/oracle/checkpoints/'

def amps_format_check(x):
    return len(x) <= MAX_LEN
    #return len(x) >= MIN_LEN and len(x) <= MAX_LEN

def preprocess_str2ids(seqs):
    ret_seqs = []
    for x in tqdm(seqs, desc="preprocess_str2ids"):
        assert amps_format_check(x)
        if IS_CONDITIONAL:
            x = ["<bos>", "<cls>"] + [c.upper() for c in x] + ["<eos>"]
        else:
            x = ["<bos>"] + [c.upper() for c in x] + ["<eos>"]
        x = x + ["<pad>"] * (PADDING_LEN - len(x))
        x = [VOCAB_STOI[c] for c in x]
        ret_seqs.append(torch.tensor(x))
    ret_seqs = torch.stack(ret_seqs)
    return ret_seqs


def get_data4oracle(data_path):

    df = pd.read_csv(data_path, names=["label", "sequence"])
    seq_ids = preprocess_str2ids(df["sequence"])
    X_ids = seq_ids
    y = torch.tensor(df["label"].to_numpy())

    return X_ids, y 

# =============  DO NOT CHANGE ================

def train_transformer_MCdropout():

    oracle_data_path = "./data/unbalanced_dataset_minus_test.csv"
    compare_data_path = "./data/unbalanced_dataset_test.csv"

    
    # get training data for oracle
    X_ids_train, y_train = get_data4oracle(oracle_data_path)
    # get data for performance comparison 
    X_ids_test, y_test = get_data4oracle(compare_data_path)

    X_ids_train, y_train = X_ids_train.numpy(), y_train.numpy()
    X_ids_test, y_test = X_ids_test.numpy(), y_test.numpy()


    
