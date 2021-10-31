import argparse
import datetime
import random 
from argparse import Namespace
import numpy as np
import torch
from torch.backends import cudnn

from rl_finetune import rlft_train 
#from eval_pipeline import accuracy

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='../tmp/amp_basic/rlft')
    parser.add_argument('--userinfo', type=str, default="")

    parser.add_argument('--device', type=str,
        default='cuda:7' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--oracle-type', type=str, default='RandomForest')
    parser.add_argument('--feature-type', type=str, default='CTDD')

    args_main = parser.parse_known_args()[0]
    args_train = rlft_train.get_args()
    args = Namespace(**vars(args_main), **vars(args_train))

    # Log time 
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    args.t0 = t0
    return args 


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True 

def main():
    torch.set_num_threads(1) # for poor CPU

    args = get_args()
    print("args: ", args, "\n")

    set_seed(args.seed)

    rlft_train.train(args)


if __name__ == "__main__":
    main()