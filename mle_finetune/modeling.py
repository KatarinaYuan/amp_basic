import argparse 
import torch 
import torch.nn as nn 
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    #AutoModelForSeq2SeqLM, 
    #BartForConditionalGeneration,
    #CONFIG_MAPPING, 
    #MODEL_MAPPING
    XLNetTokenizer,
    XLNetConfig,
    XLNetLMHeadModel,
)

#MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
#MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name-or-path', type=str, required=True)
    parser.add_argument('--config-name', type=str, default=None)
    parser.add_argument('--tokenizer-name', type=str, default=None)
    parser.add_argument('--use-slow-tokenizer', action='store_true', default=False,
            help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    #parser.add_argument('--model-type', type=str, default=None, 
    #        choices=MODEL_TYPES) # NOTE

    args = parser.parse_known_args()[0]

    return args

def fix_encoder(model, set_grad=False, tie_embedding=True):
    return 
    if set_grad:
        model.transformer.requires_grad_(False) # For XLNet
        if tie_embedding:
            model.transformer.word_embedding.requires_grad_(True)
        else:
            wt = model.lm_loss.weight.data.clone().detach()
            bs = model.lm_loss.bias.data.clone().detach()
            model.lm_loss = nn.Linear(model.config.d_model, model.config.vocab_size, bias=True)
            with torch.no_grad():
                model.lm_loss.weight.data.copy_(wt)
                model.lm_loss.bias.data.copy_(bs)

    # For dropout
    if isinstance(model, XLNetLMHeadModel):
        model.transformer.eval()
    else:
        model.module.transformer.eval()


def build_tokenizer_model(args, print_func):
    """ Load pretrained model and tokenizer """

    print("config: ", args.config_name, args.model_name_or_path, args.tokenizer_name)

    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = XLNetConfig()
        print_func("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = XLNetTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = XLNetLMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        print_func("Training new model from scratch")
        model = XLNetLMHeadModel.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    fix_encoder(model, set_grad=True)

    return tokenizer, model 