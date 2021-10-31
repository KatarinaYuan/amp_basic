import time
import argparse
import os
import datetime 
import ipdb

from argparse import Namespace
from torch.backends import cudnn
from transformers import set_seed

from oracle import sklearn_train, nn_train, read_data, transform_input
from ood_eval import ood_estimate


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

    parser.add_argument('--oracle-type', type=str)
    parser.add_argument('--train-file', type=str, default='./data/amp_pos_spaced_train.csv')
    parser.add_argument('--test-file', type=str, default='./data/amp_pos_spaced_test.csv')

    # Feature
    parser.add_argument('--feature-type', type=str, default='CTDD')

    args_run = parser.parse_known_args()[0]

    #if args_run.oracle_type in ['KNN', 'RandomForest']:
    #    args_train = sklearn_train.get_args()
    #elif args_run.oracle_type in ['MLP', 'LSTM', 'Transformer']:
    #    args_train = nn_train.get_args()
    #else:
    #    raise NotImplementedError
    
    args = Namespace(**vars(args_run), **vars(sklearn_train.get_args()), **vars(nn_train.get_args()))

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
    raw_datasets = read_data(args.train_file, args.test_file, seed=args.seed)
    preprocessed_datasets = transform_input(args, raw_datasets)

    ipdb.set_trace()
    prob_dict = {}
    pred_dict = {}
    label_dict = {}
    for oracle_type in ['KNN', 'RandomForest', 'Transformer', 'LSTM']:
        args.oracle_type = oracle_type
        if args.oracle_type in ['KNN', 'RandomForest']:
            preprocessed_datasets.set_format(type='numpy', columns=['label', f'feat_{args.feature_type}'])
            #res_train = sklearn_train.train(args, model, preprocessed_datasets)
            res_test = sklearn_train.test(args, preprocessed_datasets['test'])
        else:
            if args.oracle_type == 'LSTM':
                args.num_layers = 2
            else:
                args.num_layers = 1
            preprocessed_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
            #res_train = nn_train.train(args, model, preprocessed_datasets)
            res_test = nn_train.test(args, preprocessed_datasets['test'])
        
        prob_dict[oracle_type] = res_test[2]
        pred_dict[oracle_type] = res_test[3]
        label_dict[oracle_type] = preprocessed_datasets['test']['label']
    
    ood_estimate.calibration(prob_dict, label_dict)
    ipdb.set_trace()
    
    print("Done. tm: {} record_prefix: {}".format(time.time()-beg_tm, args.beg_tm))


if __name__ == '__main__':
    main()
