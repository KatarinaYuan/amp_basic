import argparse
from argparse import Namespace

import ipdb

import datasets
from datasets import load_dataset
import pandas as pd 
import numpy as np 
import random
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator, DataCollatorForLanguageModeling, DataCollatorForPermutationLanguageModeling


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--dataset-config-name', type=str, default=None)
    parser.add_argument('--train-file', type=str, default=None) # csv file
    parser.add_argument('--validation-file', type=str, default=None) # csv file
    parser.add_argument('--validation-split-percentage', default=5) # NOTE
    # format
    parser.add_argument('--pad-to-max-length', action='store_true', 
            help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument('--max-seq-length', type=int, default=512, # NOTE
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.")
    parser.add_argument('--line-by-line', default=False, action='store_true',
            help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.")
    # data loader
    parser.add_argument('--per-device-train-batch-size', type=int, default=3)
    parser.add_argument('--per-device-eval-batch-size', type=int, default=1)
    parser.add_argument('--plm-probability', type=float, default=1 / 6, 
            help="Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling.")
    parser.add_argument('--max-span-length', type=int, default=5,
            help="Maximum length of a span of masked tokens for permutation language modeling.")

    args = parser.parse_known_args()[0]

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "txt"]
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "txt"]

    return args

def download_dataset(args):
    if args.dataset_name is not None:
        # In distributed training, the load_dataset function guarantee that only one 
        # local process can concurrently download the dataset.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        
        # Split
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)

        # Split
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
            )
        
    if args.reduce_data:
        # NOTE: to reduce data amount
        for typ in ['train', 'validation']:
            idx = np.random.choice(raw_datasets[typ].num_rows, raw_datasets[typ].num_rows//args.reduce_data, replace=False)
            raw_datasets[typ] = datasets.Dataset.from_dict(raw_datasets[typ][idx])
        
    return raw_datasets 

def plm_preprocess(args, raw_datasets, tokenizer, print_func):
    """ Preprocessing the datasets. """

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Max Length
    if args.max_seq_length > tokenizer.model_max_length:
        print_func(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    
    # Line by line
    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line.strip()+'</s>' for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset line_by_line",
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])
        
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

        #et_trace()

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result
        
        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

        #ipdb.set_trace()

    print("[plm_preprocess]: tokenized_datasets: ", tokenized_datasets['train'][0])

    return tokenized_datasets



def build_dataset_dataloader(args, tokenizer, model, print_func, use_fp16=False):
    # Datasets
    raw_datasets = download_dataset(args)
    processed_datasets = plm_preprocess(args, raw_datasets, tokenizer, print_func)
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    print_func("[build_dataset_dataloader]: ", processed_datasets)

    #ipdb.set_trace()

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print_func(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer,
        plm_probability=args.plm_probability,
        max_span_length=args.max_span_length,
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    for b in train_dataloader:
        print_func("[build_dataset_dataloader]: train_dataloader: train_batch: ", b)
        break

    return (raw_datasets, train_dataset, eval_dataset), (train_dataloader, eval_dataloader)


