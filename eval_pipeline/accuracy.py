import ipdb
import time
import argparse
import os
import datetime

from argparse import Namespace

import torch 
import numpy as np 

from transformers import XLNetTokenizer

from feature.CTDD import CTDD_array
from feature.Prottrans import Prottrans_array

from oracle import sklearn_train, nn_train, dataset

oracle_args = None 
oracle_tokenizer = None 
oracle_model = None


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--output-dir', type=str, default=None)
    #parser.add_argument('--verbo', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--oracle-type', type=str)
    parser.add_argument('--feature-type', type=str, default='CTDD')

    args_run = parser.parse_known_args()[0]
    
    args = Namespace(**vars(args_run), **vars(sklearn_train.get_args()), **vars(nn_train.get_args()))

    #if args.output_dir is not None:
        ## args.output_dir = os.path.join(args.output_dir, args.model_name_or_path, args.train_file.split('.')[0].split('_')[0], f"seed_{args.seed}")
    #    os.makedirs(args.output_dir, exist_ok=True)
    #args.beg_tm = datetime.datetime.now().strftime('%m%d_%H%M%S')

    print("oracle_args:", args)
    return args 

def _get_global_oracle(args):

    global oracle_args
    if oracle_args is None:
        oracle_args = get_args()
        oracle_args.oracle_type = args.oracle_type
        oracle_args.feature_type = args.feature_type
        oracle_args.device = args.device
    
    global oracle_model
    if oracle_model is None:
        if oracle_args.oracle_type in ['KNN', 'RandomForest']:
            load_file = os.path.join(sklearn_train.SAVE_PATH, oracle_args.feature_type, 'sklearn', f'{oracle_args.oracle_type}.pkl')
            oracle_model = sklearn_train.load_model(load_file)
        else:
            if oracle_args.oracle_type == 'LSTM':
                oracle_args.num_layers = 2
            else:
                oracle_args.num_layers = 1
            load_file = os.path.join(nn_train.SAVE_PATH, oracle_args.feature_type, 'nn', f'{oracle_args.oracle_type}.ckpt')
            oracle_model = nn_train.build_model(oracle_args)
            oracle_model = nn_train.load_model(oracle_model, load_file)
    
    global oracle_tokenizer 
    if oracle_tokenizer is None:
        oracle_tokenizer = XLNetTokenizer.from_pretrained('Rostlab/prot_xlnet')



def cal_amplike_probability(args, seq):
    '''
    Params:
        seq: a string
    '''
    
    _get_global_oracle(args)
    global oracle_args
    global oracle_model
    global oracle_tokenizer
    
    seq = seq.strip()
    # Transform data
    x_data = oracle_tokenizer(
        seq,
        padding='max_length',
        truncation=True,
        max_length=dataset.MAX_SEQ_LENGTH,
    )
    x_data = dict(x_data)
    #ipdb.set_trace()
    if args.feature_type == 'Prottrans':
        x_data['feat_Prottrans'] = Prottrans_array([seq], 'T5', oracle_args.device, bsz=258)
    else:
        x_data['feat_CTDD'] = CTDD_array([seq])
    
    if oracle_args.oracle_type in ['KNN', 'RandomForest']:
        #x_data.set_format(type='numpy', columns=['label', f'feat_{oracle_args.feature_type}'])
        for key in [f'feat_{oracle_args.feature_type}']:
            x_data[key] = np.array(x_data[key])
        
        test_x = x_data["feat_"+args.feature_type]
        b_pred = oracle_model.predict(test_x)
        b_prob = oracle_model.predict_proba(test_x)
        
    else:
        #x_dat.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        for key in ['input_ids', 'attention_mask']:
            x_data[key] = torch.tensor(x_data[key], device=oracle_args.device)
        # Pass through oracle model
        with torch.no_grad():
            logits = oracle_model(x_data['input_ids'], x_data['attention_mask'])
            b_prob, b_pred = oracle_model.predict(logits)
    return b_prob





        

            
        
