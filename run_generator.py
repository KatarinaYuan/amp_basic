import ipdb 
import datetime
import argparse
import os  
from tqdm import tqdm 
import pandas as pd 

from torch.backends import cudnn
from transformers import set_seed
from transformers import TextGenerationPipeline, XLNetTokenizer, XLNetLMHeadModel

from oracle import read_data, transform_input
from generation.generator import get_generator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--verbo', type=int, default=10)
    parser.add_argument('--userinfo', type=str, default='')
    parser.add_argument('--overwrite-cache', default=False, action="store_true",
            help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--preprocessing-num-workers', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:3')

    parser.add_argument('--train-file', type=str, default='./data/spaced_train_pos_data.csv')
    parser.add_argument('--test-file', type=str, default='./data/spaced_test_pos_data.csv')

    parser.add_argument('--content-size', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--num-return-sequences', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=62)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=3)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--ith', type=int, default=0)

    args = parser.parse_known_args()[0]

    #args.device = 'cuda:{}'.format(args.ith)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    args.beg_tm = datetime.datetime.now().strftime('%m%d_%H%M%S')

    print("args:", args)
    return args 

if __name__ == '__main__':
    args = get_args()

    if args.seed is not None:
        set_seed(args.seed)
        cudnn.deterministic = True ## NOTE
        cudnn.benchmark = False 

    train_file = 'spaced_train_pos_data.csv'
    test_file = 'spaced_test_pos_data.csv'
    raw_datasets = read_data(train_file, test_file, seed=args.seed)
    def trunc_function(examples):
        examples['text'] = [line[:args.content_size*2-1] for line in examples['text']] 
        return examples 
    trunc_datasets = raw_datasets.map(
        trunc_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Truncate sequence",
    )
    #preprocessed_datasets = transform_input(args, trunc_datasets, max_length=args.content_size)
    #preprocessed_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    #ipdb.set_trace()

    path = '/tmp/test-plm/Rostlab/prot_xlnet/seed_0/0925_215606/'

    tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
    model = XLNetLMHeadModel.from_pretrained(path).to(args.device)
    model.requires_grad_(False)
    model.eval()

    if args.device == 'cpu':
        device_id = -1
    else:
        device_id = eval(args.device.split(':')[-1])
    pipe = TextGenerationPipeline(model, tokenizer, device=device_id)
    text_inputs = trunc_datasets['train']['text'] + trunc_datasets['validation']['text'] + trunc_datasets['test']['text']
    
    comp_res, res = [], []
    #for i in tqdm(range(len(text_inputs))):
    sz = len(text_inputs) // args.K
    l = args.ith * sz 
    r = min((args.ith + 1) *sz,  len(text_inputs))
    for i in tqdm(range(l, r)):
        o = pipe(text_inputs[i], do_sample=True, 
                num_return_sequences=args.num_return_sequences, 
                temperature=args.temperature,
                #top_k=args.top_k,
                #top_p=args.top_p,
                num_beams=args.num_beams, 
                #repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_length=args.max_length)
        o = [x['generated_text'] for x in o]
        comp_o = ["".join(x.split()) for x in o]
        o = ["".join(x.split('▁')) for x in comp_o]
        comp_res.extend(comp_o)
        res.extend(o)
        #if i % 10 == 0:
        #    ipdb.set_trace()
    df = pd.DataFrame(res, columns=['text'])
    ipdb.set_trace()
    df.to_csv(f'./output/beam:{args.num_beams}-total:{len(res)}-ith:{args.ith}-ctx:{args.content_size}-ngram:{args.no_repeat_ngram_size}-device:{args.device}.csv')
    ipdb.set_trace()

    '''G = get_generator('model', 1, 'cuda:1', model_name_or_path=path)
    ipdb.set_trace()
    seqs = G.generate_batch(preprocessed_datasets['test'])
    ipdb.set_trace()'''