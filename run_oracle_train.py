import time
import argparse
import os
import datetime 
import ipdb

from argparse import Namespace
from torch.backends import cudnn
from transformers import set_seed

from oracle import sklearn_train, nn_train, read_data, transform_input


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--verbo', type=int, default=100)
    parser.add_argument('--userinfo', type=str, default='')
    parser.add_argument('--overwrite-cache', default=False, action="store_true",
            help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--preprocessing-num-workers', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--oracle-type', type=str, required=True)
    parser.add_argument('--train-file', type=str, default='./data/toy_train_data.csv')
    parser.add_argument('--test-file', type=str, default='./data/toy_test_data.csv')

    # Feature
    parser.add_argument('--feature-type', type=str, default='CTDD')

    args_run = parser.parse_known_args()[0]

    if args_run.oracle_type in ['KNN', 'RandomForest']:
        args_train = sklearn_train.get_args()
    elif args_run.oracle_type in ['MLP', 'LSTM', 'Transformer']:
        args_train = nn_train.get_args()
    else:
        raise NotImplementedError
    
    args = Namespace(**vars(args_run), **vars(args_train))

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

    if args.oracle_type in ['KNN', 'RandomForest']:
        model = sklearn_train.build_model(args)
    else:
        model = nn_train.build_model(args).to(args.device)
    raw_datasets = read_data(args.train_file, args.test_file, seed=args.seed)
    preprocessed_datasets = transform_input(args, raw_datasets)

    #ipdb.set_trace()
    if args.oracle_type in ['KNN', 'RandomForest']:
        preprocessed_datasets.set_format(type='numpy', columns=['label', f'feat_{args.feature_type}'])
        res_train = sklearn_train.train(args, model, preprocessed_datasets)
        res_test = sklearn_train.test(args, preprocessed_datasets['test'])
    else:
        preprocessed_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        res_train = nn_train.train(args, model, preprocessed_datasets)
        res_test = nn_train.test(args, preprocessed_datasets['test'])
    
    print("Done. tm: {} record_prefix: {}".format(time.time()-beg_tm, args.beg_tm))


if __name__ == '__main__':
    main()
