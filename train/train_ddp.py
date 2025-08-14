import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import shutil
import wandb
import json
import pickle
import torch
from torch.distributed import init_process_group, destroy_process_group

from datetime import datetime
from utils.utils import create_folder_if_not_exist, create_folder_overwrite_if_exist, clean_print
from data.data import AutoRegressDataset, AdditionDataset, AdditionTokenizer, MultiplicationDataset, MultiplicationTokenizer
from train.trainer import Trainer
from configs import get_experiment_config
import setproctitle
setproctitle.setproctitle("CleanGPT@Debug")

def ddp_setup():
    num_cores = os.cpu_count()
    num_threads = max(1, min(4, num_cores // 4))    # Each process uses part of the CPU cores
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MASTER_ADDR"] = "localhost"         # localhost for single node
    os.environ["MASTER_PORT"] = "21662"             # Any free port
    init_process_group(backend="nccl")              # nccl for linux, gloo for windows
    torch.cuda.set_device(int(os.environ.get("RANK", default='0')))

def get_args_ready(experiment_name:str, RANK:int):
    ''' train a miniature character-level shakespeare model, good for debugging and playing on macbooks and such '''
    args = get_experiment_config(experiment_name)
    clean_print(f'Exp Profile: {args.exp_profile}', RANK, '[Trainer]')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = f'{base_path}/out/{args.exp_name}/{args.exp_profile}/{timestamp}'
    
    # assert some hyper paras
    assert args.dataset in ['tinystory', 'shakespeare_char', 'adder', 'multiplier'], f"dataset {args.dataset} not supported"
    assert args.train_iters % args.eval_interval == 0
    assert args.train_iters % args.save_interval == 0

    # get ready for wandb logging
    if RANK == 0:
        create_folder_if_not_exist(f'{base_path}/Wandb')
    if not args.wandb:       
        os.environ['WANDB_MODE'] = 'offline'

    return args

def load_dataset(args):
    if args.dataset == 'tinystory':
        dataset_train = AutoRegressDataset(args, f'{base_path}/data/tinystory/train.npy')
        dataset_val = AutoRegressDataset(args, f'{base_path}/data/tinystory/val.npy')
        dataset_test = None
        with open(os.path.join(f'{base_path}/data/{args.dataset}/meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            tokenizer = meta['tokenizer']
            args.vocab_size = tokenizer.vocab_size
    elif args.dataset == 'shakespeare_char':
        dataset_train = AutoRegressDataset(args, f'{base_path}/data/shakespeare_char/train.npy')
        dataset_val = AutoRegressDataset(args, f'{base_path}/data/shakespeare_char/val.npy')
        dataset_test = None
        with open(os.path.join(f'{base_path}/data/{args.dataset}/meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            tokenizer = None
            args.vocab_size = meta['vocab_size']
    elif args.dataset == 'adder':
        dataset_train = AdditionDataset(args.adder_ndigit, 'train', format_vocab=args.adder_format_vocab)
        dataset_val = AdditionDataset(args.adder_ndigit, 'val', format_vocab=args.adder_format_vocab)
        dataset_test = AdditionDataset(args.adder_ndigit, 'test', format_vocab=args.adder_format_vocab)
        tokenizer = AdditionTokenizer(args.adder_ndigit, format_vocab=args.adder_format_vocab)
        args.vocab_size = 10 + len(args.math_vocab) if args.adder_use_format else 10
    elif args.dataset == 'multiplier':
        dataset_train = MultiplicationDataset(args.adder_ndigit, 'train', format_vocab=args.multiplier_format_vocab)
        dataset_val = MultiplicationDataset(args.adder_ndigit, 'val', format_vocab=args.multiplier_format_vocab)
        dataset_test = MultiplicationDataset(args.adder_ndigit, 'test', format_vocab=args.multiplier_format_vocab)
        tokenizer = MultiplicationTokenizer(args.adder_ndigit, format_vocab=args.multiplier_format_vocab)
        args.vocab_size = 10 + len(args.math_vocab) if args.adder_use_format else 10
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
    
    dataset_dict = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    return dataset_dict, tokenizer

def save_setting(args):
    if (args.save_ckpt or args.save_snapshot) and RANK == 0:
        # create floder to save ckpts and hyperparas if we need
        create_folder_if_not_exist(f'{args.out_dir}/{args.save_strategy}')
        with open(f'{args.out_dir}/config.json', 'w') as f:
            f.write(json.dumps(vars(args), indent=4))
        
        # save the training script
        script_path = os.path.abspath(__file__)
        shutil.copy2(
            src=script_path,
            dst=f"{args.out_dir}/train_script.py",
        )    

if __name__ == "__main__":
    # init DDP process group
    ddp_setup()
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", default='1'))
    RANK = int(os.environ.get("RANK", default='0'))

    # activate tf32 on matmul and cudnn to boost NVIDIA Ampere GPU performance
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    
    # get hyper paras ready
    experiment_name = 'TinyStory_Llama'
    args = get_args_ready(experiment_name, RANK)

    # build training objs
    dataset_dict, tokenizer = load_dataset(args)

    # save setting
    save_setting(args)

    # train
    for seed in args.seeds:
        if args.save_ckpt and args.save_strategy == 'interval' and RANK == 0:
            create_folder_overwrite_if_exist(f'{args.out_dir}/interval/{seed}')

        # This unique id is necessary for log resuming
        wandb_id = wandb.util.generate_id() 
        
        # build trainer
        trianer = Trainer(args, seed, wandb_id, dataset_dict, tokenizer)

        # wandb log only on rank0
        if RANK == 0:
            with wandb.init(
                project=args.wandb_project,
                group = args.exp_profile,
                name = f"seed_{seed}",
                id = trianer.wandb_id,
                resume = 'allow',
                dir = f'{base_path}/Wandb',
                config=args
            ):
                raw_model = trianer.model.module if hasattr(trianer.model, "module") else trianer.model
                wandb.watch(raw_model, log='all', log_freq=100)
                trianer.train()
        else:
            trianer.train()

        wandb.finish()
        assert wandb.run is None

    # destroy DDP process group
    destroy_process_group()