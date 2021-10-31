import time
import argparse
import os
import datetime 
import ipdb
from tqdm import tqdm 

from argparse import Namespace
from torch.backends import cudnn
from transformers import set_seed

from oracle import sklearn_train, nn_train, read_data, transform_input
from ood_eval import ood_estimate
from eval_pipeline import accuracy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--verbo', type=int, default=100)
    parser.add_argument('--userinfo', type=str, default='')
    #parser.add_argument('--overwrite-cache', default=False, action="store_true",
    #        help="Overwrite the cached training and evaluation sets")
    #parser.add_argument('--preprocessing-num-workers', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--oracle-type', type=str, default='RandomForest')
    #parser.add_argument('--train-file', type=str, default='./data/amp_pos_spaced_train.csv')
    #parser.add_argument('--test-file', type=str, default='./data/amp_pos_spaced_test.csv')

    # Feature
    parser.add_argument('--feature-type', type=str, default='CTDD')

    args_run = parser.parse_known_args()[0]

    #if args_run.oracle_type in ['KNN', 'RandomForest']:
    #    args_train = sklearn_train.get_args()
    #elif args_run.oracle_type in ['MLP', 'LSTM', 'Transformer']:
    #    args_train = nn_train.get_args()
    #else:
    #    raise NotImplementedError
    
    #args = Namespace(**vars(args_run), **vars(sklearn_train.get_args()), **vars(nn_train.get_args()))
    args = args_run

    if args.output_dir is not None:
        ## args.output_dir = os.path.join(args.output_dir, args.model_name_or_path, args.train_file.split('.')[0].split('_')[0], f"seed_{args.seed}")
        os.makedirs(args.output_dir, exist_ok=True)
    args.beg_tm = datetime.datetime.now().strftime('%m%d_%H%M%S')

    print("args:", args)
    return args 

def main():
    beg_tm = time.time()

    args = get_args()
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        cudnn.deterministic = True ## NOTE
        cudnn.benchmark = False 

    #if args.oracle_type in ['KNN', 'RandomForest']:
    #    model = sklearn_train.build_model(args)
    #else:
    #    model = nn_train.build_model(args).to(args.device)
    
    #raw_datasets = read_data(args.train_file, args.test_file, seed=args.seed)
    #preprocessed_datasets = transform_input(args, raw_datasets)
    x = 'M▁S▁R▁R▁G▁T▁A▁E▁E▁K▁T▁A▁K▁S▁D▁P▁I▁Y▁R▁N▁R▁L▁V▁N▁M▁L▁V▁N▁R▁I▁L▁K▁H▁G▁K▁K▁S▁L▁A▁Y▁Q▁I▁I▁Y▁R▁A▁L▁K▁K▁I▁Q▁Q▁K▁T▁E▁T▁N▁P▁L▁S'
    x = ['A▁A▁A▁A▁G▁S▁C▁V▁W▁G▁A▁V▁N▁Y▁T▁S▁D▁C▁A▁A▁E▁C▁K▁R▁R▁G▁Y▁K▁G▁G▁H▁C▁G▁S▁F▁A▁N▁V▁N▁C▁W▁C▁E▁T'
        ,'A▁A▁A▁A▁G▁S▁C▁V▁W▁G▁A▁V▁N▁Y▁T▁S▁D▁C▁N▁G▁E▁C▁K▁R▁R▁G▁Y▁K▁G▁G▁H▁C▁G▁S▁F▁A▁N▁V▁N▁C▁W▁C▁E▁T'
        ,'A▁A▁A▁A▁G▁S▁C▁V▁W▁G▁A▁V▁N▁Y▁T▁S▁D▁C▁N▁G▁E▁C▁L▁L▁R▁G▁Y▁K▁G▁G▁H▁C▁G▁S▁F▁A▁N▁V▁N▁C▁W▁C▁R▁T'
        ,'A▁A▁A▁A▁L▁S▁R▁A▁A▁L▁R▁A▁A▁V▁A'
        ,'A▁A▁A▁A▁L▁S▁R▁W▁W▁L▁R▁W▁W▁V▁A'
        ,'A▁A▁C▁S▁L▁G▁S▁L▁L▁N▁V▁G▁C▁N▁S▁C▁A▁C▁A▁A▁H▁C▁L▁A▁T▁R▁G▁K▁N▁G▁A▁C▁N▁S▁Q▁R▁R▁C▁V▁C▁N▁K'
        ,'A▁A▁F▁R▁G▁C▁W▁T▁K▁S▁Y▁S▁P▁K▁P▁C▁L▁G▁K▁R'
        ,'A▁A▁G▁G▁V▁K▁K▁P▁K▁K▁A▁A▁A▁A▁K▁K▁S▁P▁K▁K▁P▁K▁K▁P▁A▁A▁A'
        ,'A▁A▁I▁Y▁P▁F▁G▁I▁K▁I▁R▁C▁K▁A▁A▁F▁C'
        ,'A▁A▁L▁K▁G▁C▁W▁T▁K▁S▁I▁P▁P▁K▁P▁C▁F▁G▁F'
        ,'A▁A▁L▁K▁G▁C▁W▁T▁K▁S▁I▁P▁P▁K▁P▁C▁F▁G▁K▁R'
        ,'A▁A▁L▁R▁G▁C▁W▁T▁K▁S▁I▁P▁P▁K▁P▁C▁P▁G▁K▁R'
        ,'A▁A▁L▁R▁G▁C▁W▁T▁K▁S▁I▁P▁P▁K▁P▁C▁S▁G▁K▁R'
        ,'A▁A▁L▁S▁E▁L▁H▁C▁D▁K▁L▁H▁V▁D▁P▁E▁N▁F▁K▁L▁L'
        ,'A▁A▁N▁F▁G▁P▁S▁V▁F▁T▁P▁E▁V▁H▁E▁T▁W▁Q▁K▁F▁L▁N▁V▁V▁V▁A▁A▁L▁G▁K▁Q▁Y▁H'
        ,'A▁A▁N▁I▁P▁F▁K▁V▁H▁F▁R▁C▁K▁A▁A▁F▁C'
        ,'A▁A▁N▁I▁P▁F▁K▁V▁H▁F▁R▁C▁K▁S▁I▁F▁C'
        ,'A▁A▁P▁R▁G▁G▁K▁G▁F▁F▁C▁K▁L▁F▁K▁D▁C']
    for _ in tqdm(x):
        prob = accuracy.cal_amplike_probability(args, _)
        #ipdb.set_trace()
    ipdb.set_trace()
    
    ipdb.set_trace()
    
    print("Done. tm: {} record_prefix: {}".format(time.time()-beg_tm, args.beg_tm))


if __name__ == '__main__':
    main()
