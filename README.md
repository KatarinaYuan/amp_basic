# amp_basic
---- 
##  Environment
> python==3.7.11
> pytorch==1.8.2
> transformers==4.10.2
> accelerate==0.4.0
> numpy 
> pandas


Please refer to `env.yaml` to get more detailed library requirements

### To test the environment
Run a small fraction of data. Add `--reduce-data {K}` after the command to train $$\frac{1}{K}$$ data in order to test the environment.

## Introduction
This is a repo doing language modeling task for permutation language models (e.g. XLNet). It imitates [`run_mlm_no_trainer.py`](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)
to give a MLE (maximum likelihood estimates) training paradigm for AMP (anti-microbial peptides) generation. It uses `huggingface.transformers` as foundation and avoids need of the abstract function `huggingface.transformers.trainer`
which allows us to control the training process.

## Usage 

### Data                                                                         
Put `train_data.csv` and `valid_data.csv` under `./data/`. Make sure its column_names =     ['label', 'text']. `text` contains a sequence composed of alphabet characters (e.g. 'A',     'M'), and `label` indicates whether this sequence is an AMP

### Single-GPU
```
python run_mleft.py --train-file ./data/train_data.csv --test-file ./data/test_data.csv --model-name-or-path Rostlab/prot_xlnet --do-train --do-eval --output-dir /tmp/test-plm --overwrite-cache --tokenizer-name Rostlab/prot_xlnet --pad-to-max-length --max-seq-length 62 --line-by-line --reduce-data 100
```

### Multi-GPU
Use `huggingface.accelerate`. Refer to [guidance](https://github.com/huggingface/transformers/tree/master/examples/pytorch) for installation.

Replace 'python' with 'accelerate launch' to run the file. 
i.e.
```
accelerate launch path_to_script.py --args_to_script
```

## NOTE

Now, to run `--reduce-data 100` requires around 5 minutes and 3185MiB on GPU
