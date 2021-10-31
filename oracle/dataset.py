import ipdb
import time

import torch 

import datasets
from datasets import load_dataset
from transformers import XLNetTokenizer

from feature.CTDD import CTDD_array
from feature.Prottrans import Prottrans_array

# VOCAB_LIST = ['</s>', '▁A', '▁L', '▁G', '▁V', '▁S', '▁R', '▁E', '▁D', '▁T', '▁I', '▁P', '▁K', '▁F', '▁Q', '▁N', '▁Y', '▁M', '▁H', '▁W', '▁C', '▁X', '▁B', '▁O', '▁U', '▁Z']
VOCAB_SIZE = 37 # 
MAX_SEQ_LENGTH = 62 # plus '</s>'

from transformers import XLNetLMHeadModel, XLNetTokenizer, pipeline

"""
tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
PreTrainedTokenizer(
		name_or_path='Rostlab/prot_xlnet', 
		vocab_size=37, 
		model_max_len=1000000000000000019884624838656, 
		is_fast=False, 
		padding_side='left', 
		special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 
										'unk_token': '<unk>', 'sep_token': '<sep>', 
										'pad_token': '<pad>', 'cls_token': '<cls>', 
										'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True), 
										'additional_special_tokens': ['<eop>', '<eod>']})
tokenizer.convert_ids_to_tokens([x for x in range(0,37)])
['X', '<s>', '</s>', '<cls>', '<sep>', '<pad>', '<mask>', '<eod>', '<eop>', '.', 
'(', ')', '"', '-', '–', '£', '€', 
'▁L', '▁S', '▁A', '▁G', '▁E', '▁V', '▁T', '▁R', '▁D', '▁I', 
'▁P', '▁K', '▁N', '▁F', '▁Q', '▁Y', '▁H', '▁M', '▁C', '▁W']
"""


def read_data(train_file, test_file, validation_split_percentage=10, seed=0):

    data_files = {'train': train_file, 'test': test_file}
    extension = train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)
    raw_datasets = raw_datasets.shuffle(seed=seed)
    num = raw_datasets['train'].num_rows * validation_split_percentage // 100
    # Split
    raw_datasets["validation"] = datasets.Dataset.from_dict(raw_datasets['train'][:num])
    raw_datasets["train"] = datasets.Dataset.from_dict(raw_datasets['train'][num:])

    return raw_datasets

def transform_input(args, raw_datasets, max_length=MAX_SEQ_LENGTH):
    text_column_name = 'text'
    tokenizer = XLNetTokenizer.from_pretrained('Rostlab/prot_xlnet')

    # Get input_ids
    def tokenize_function(examples):
        examples[text_column_name] = [line.strip() for line in examples[text_column_name]] # if len(line) > 0 and not line.isspace()], # NOTE: guarantee no empty lines
        return tokenizer(
            examples[text_column_name],
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset line_by_line",
    )


    # Get feat_embed

    def preprocessed_function_CTDD(examples):
        return {'feat_CTDD': CTDD_array(examples[text_column_name]),}
    def preprocessed_function_Prottrans(examples):
        return {'feat_Prottrans': Prottrans_array(examples[text_column_name], 'T5', args.device, bsz=258)}
        
    beg_tm = time.time()
    preprocessed_datasets = tokenized_datasets.map(
        preprocessed_function_Prottrans if args.feature_type == 'Prottrans' else preprocessed_function_CTDD,
        batched=True,
        batch_size=len(tokenized_datasets['train']),
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running feature extractor on dataset",
    )
    print("time needed: ", time.time()-beg_tm)
    #ipdb.set_trace()

    return preprocessed_datasets
