'''
Credit: https://github.com/clamp-gen/common-evaluation/blob/main/clamp_common_eval/oracles/PortTrans.py
'''

import math
import re
import pdb
import torch
import ipdb
import ipdb

from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AlbertTokenizer
import gc
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np
import time 

import transformers 
transformers.logging.set_verbosity_error()

ProttransTokenizer = None 
ProttransModel = None 

def Prottrans_loader(model_name, device='cuda:0'):
    beg_tm = time.time()
    global ProttransTokenizer
    global ProttransModel

    if ProttransModel is not None:
        return ProttransTokenizer, ProttransModel

    if model_name == "T5":
        model_name = "Rostlab/prot_t5_xl_uniref50" 
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_name)
    elif model_name == "AlBert":
        model_name = "Rostlab/prot_albert" 
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AutoModel.from_pretrained(model_name)
    elif model_name == "Bert":
        model_name = "Rostlab/prot_bert" 
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = BertModel.from_pretrained(model_name)
    else:
        raise NotImplementedError

    model = model.to(device)
    model.requires_grad_(False)
    model = model.eval()
    
    ProttransTokenizer = tokenizer
    ProttransModel = model 

    print("[Prottrans_loader]: time: ", time.time()-beg_tm)
    return tokenizer, model

def embed_dataset(dataset_seqs, tokenizer, model, device, shift_left = 0, shift_right = -1):
    inputs_embedding = []

    #   for sample in tqdm(dataset_seqs):
    for sample in dataset_seqs:
        with torch.no_grad():
            ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
            # {'input_ids': , 'attention_mask': }
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            inputs_embedding.append(embedding[0].detach().cpu().numpy()[shift_left:shift_right])

    return inputs_embedding


def PortTrans_fast(sequences, tokenizer, model, device, **kw):
    ''' PortTrans feature extraction function. The default model is T5, from https://github.com/agemagician/ProtTrans
    Parameters:
        sequences : list of str, a list of str represent the amino acids
        tokenizer : transformers.tokenizer, a predefined tokenizer
        model : transformer.model, a feature extraction model
        device : torch.device(), which device will be used to extract features
    Returns:
        list, the extracted feature, shape = [num, feature_dim]
    '''
    if isinstance(sequences, str):
        sequences = [sequences]

    encodings = []
    for seq in sequences:
      split_sequence = [s for s in seq]
      
      seq_embd = embed_dataset([split_sequence], tokenizer, model, device)
      # import pdb; pdb.set_trace()

      mean_embed = seq_embd[0].mean(axis = 0).tolist()
      encodings.append(mean_embed)
    # import pdb; pdb.set_trace()
    return np.array(encodings)


def Prottrans_array(sequences, model_name, device, bsz=32, **kw):
    ''' PortTrans feature extraction function. 
        The default model is T5, from https://github.com/agemagician/ProtTrans

        Parameters:
            sequences : list of str, a list of str represent the amino acids
            model_name : str, use which pretrained model to extract features
            device : str, which device will be used to extract features
        Returns:
            numpy.Array, the extracted feature, shape = [num, feature_dim]
    '''
    tokenizer, model = Prottrans_loader(model_name, device)

    encodings = []
    #for seq in tqdm(sequences, desc='Prottrans feature extraction'):
        #split_sequence = [s for s in seq]
        #seq_embd = embed_dataset([split_sequence], tokenizer, model, device)
        #mean_embed = seq_embd[0].mean(axis=0).tolist()
    for i in tqdm(range(0, len(sequences), bsz), desc='Prottrans feature extraction'):
        ids = tokenizer(sequences[i:i+bsz], add_special_tokens=True, padding=True, return_tensors="pt")
        embedding = model(ids['input_ids'].to(device))[0]
        embedding = embedding[:, :-1, :].mean(dim=1) # eos_token
        seq_embed = embedding.detach().cpu().numpy()
            
        encodings.append(seq_embed)
    ret = np.concatenate(encodings, axis=0)
    print(ret.shape)
    return ret 
