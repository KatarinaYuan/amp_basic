import time 
import argparse 
import os
import datetime 
from argparse import Namespace
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter                                   
import datasets                                                                     
import transformers                                                                 
from transformers import set_seed
import accelerate
from accelerate import Accelerator
from filelock import FileLock

from mle_finetune import mleft_train, modeling, data_proc

from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preprocessing-num-workers', type=int, default=None)
    parser.add_argument('--overwrite-cache', default=False, action="store_true",
            help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--userinfo', type=str, default='')
    parser.add_argument('--reduce-data', type=int, default=0)
    args_run = parser.parse_known_args()[0]
    args_modeling = modeling.get_args()
    args_dataproc = data_proc.get_args()
    args_train = mleft_train.get_args()
    args = Namespace(**vars(args_run), **vars(args_modeling), 
                **vars(args_dataproc), **vars(args_train))
    
    # Sanity check
    if args.output_dir is not None:
        if args.dataset_name is not None:
            args.output_dir = os.path.join(args.output_dir, args.model_name_or_path, args.dataset_name, f"seed_{args.seed}")
        else:
            args.output_dir = os.path.join(args.output_dir, args.model_name_or_path, args.train_file.split('.')[0].split('_')[0], f"seed_{args.seed}")
        os.makedirs(args.output_dir, exist_ok=True)
    args.beg_tm = datetime.datetime.now().strftime('%m%d_%H%M%S')
    
    print("args:", args)
    return args 

def main():
    beg_tm = time.time()

    args = get_args()
    # Tensorboard
    log_path = os.path.join(args.output_dir, f"{args.beg_tm}" + args.userinfo)
    writer = SummaryWriter(log_path)
    writer.add_text('args', str(args))

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)])
    accelerator.print('accelerator.state: ', str(accelerator.state))

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        cudnn.deterministic = True ## NOTE
        cudnn.benchmark = False 
    
    mleft_train.train(args, accelerator, accelerator.print, writer)

    accelerator.print("Done. tm: ", time.time()-beg_tm, args.beg_tm)

if __name__ == '__main__':
    main()