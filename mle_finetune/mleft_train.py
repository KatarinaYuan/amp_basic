import argparse 
import random
import os 
import math
from tqdm.auto import tqdm
import numpy as np 
import ipdb

import torch 
from torch.utils.data.dataloader import DataLoader
from datasets import load_metric
import transformers
from accelerate import Accelerator, DistributedType
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
)

from mle_finetune import data_proc, modeling
from optimization import scheduler 


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--num-train-epochs', type=int, default=3)
    parser.add_argument('--max-train-steps', type=int, default=None,
            help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--lr-scheduler-type', type=str, default='linear',
            choices=["linear", "cosine_with_restarts", "constant"])
    parser.add_argument('--num-warmup-steps', type=int, default=0)
    parser.add_argument('--num-cycles', type=int, default=1, 
            help="For lr-scheduler-type = 'cosine_with_restarts'")

    args = parser.parse_known_args()[0]

    return args 

def build_optimizer(args, model):

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    return optimizer 

def build_lr_scheduler(args, train_dataloader, optimizer):

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = scheduler.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.num_cycles,
    )

    return lr_scheduler

def evaluate(args, model, eval_dataset, eval_dataloader, accelerator, print_func):

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]

    loss = torch.mean(losses)
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print_func(f"perplexity: {perplexity} loss: {loss}")

    return {'loss': loss, 'ppl': perplexity}


def train(args, accelerator, print_func, writer):

    tokenizer, model = modeling.build_tokenizer_model(args, print_func)
    all_datasets, (train_dataloader, eval_dataloader) = data_proc.build_dataset_dataloader(args, tokenizer, model, print_func, accelerator.use_fp16)
    optimizer = build_optimizer(args, model)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    #if accelerator.distributed_type == DistributedType.TPU:
    #    model.tie_weights()

    lr_scheduler = build_lr_scheduler(args, train_dataloader, optimizer)


    ### Training Process ###

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print_func("***** Running training *****")
    print_func(f"  Num examples = {len(all_datasets[1])}")
    print_func(f"  Num Epochs = {args.num_train_epochs}")
    print_func(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print_func(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print_func(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print_func(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    avg_loss = []
    epoch_loss = []

    intvl = args.max_train_steps // max(10, args.num_train_epochs * 2)
    print("intvl:  ", intvl)

    ##save_intermediate_steps = [math.ceil((args.max_train_steps - args.num_warmup_steps)  / args.num_cycles * k) \
    ##                            + args.num_warmup_steps for k in range(args.num_cycles)]
    ##print_func("save_intermediate_steps", save_intermediate_steps)

    for epoch in range(args.num_train_epochs):
        model.train()
        modeling.fix_encoder(model)
        for step, batch in enumerate(train_dataloader):
            #ipdb.set_trace()
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            avg_loss.append(loss.item())
            epoch_loss.append(loss.item())

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                writer.add_scalar('train/loss', np.array(avg_loss).mean(), completed_steps)
                writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], completed_steps) # two set of params
                avg_loss = []
            
            ##if lr_scheduler.state_dict()['_step_count'] in save_intermediate_steps:
            ##    cyc = save_intermediate_steps.index(lr_scheduler.state_dict()['_step_count'])
            ##    unwrapped_model = accelerator.unwrap_model(model)
            ##    unwrapped_model.save_pretrained(os.path.join(args.output_dir, args.beg_tm, f'cycle_{cyc}'), save_function=accelerator.save)

            if completed_steps % intvl == 0:
                result = evaluate(args, model, all_datasets[2], eval_dataloader, accelerator, print_func)
                for k, v in result.items():
                    writer.add_scalar(f'eval/plm/{k}', v, completed_steps)
                model.train()

            if completed_steps >= args.max_train_steps:
                break
        
        writer.add_scalar('train/epoch_loss', np.array(epoch_loss).mean(), completed_steps)
        epoch_loss = []

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(os.path.join(args.output_dir, args.beg_tm, f'epoch:{epoch}'), save_function=accelerator.save)

