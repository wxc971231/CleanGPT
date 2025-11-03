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
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))

    # Prefer IP to avoid DNS resolution issues; only set if not already provided by torchrun
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "21662")

    # Choose proper backend: nccl on linux with GPU, otherwise gloo
    backend = "nccl" if torch.cuda.is_available() and sys.platform.startswith("linux") else "gloo"
    
    # Hint socket interface for single-node runs to suppress hostname resolution warnings
    if backend == "nccl":
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")  # disable InfiniBand if not present
    else:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    # Use LOCAL_RANK if available (set by torchrun), fallback to RANK
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    # Initialize process group with explicit device_id to avoid barrier() warning
    # Older PyTorch may not support device_id; gracefully fallback
    try:
        init_process_group(backend=backend, device_id=local_rank)
    except TypeError:
        init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


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
    
    try:
        # Prefer new TF32 control APIs (PyTorch >= 2.9) with fallback for older versions
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.fp32_precision = "tf32"
    except Exception:
        # Fallback for older torch: use deprecated allow_tf32 flags if present
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    
    # get hyper paras ready
    experiment_name = 'TinyStory_Llama'
    args = get_args_ready(experiment_name, RANK)

    # custom setting
    args.wandb_project = 'CleanGPT'
    args.wandb = False
    args.save_snapshot = False                  # save the latest traing snapshot, from which we can resume training 
    args.save_ckpt = False                      # update ckpt by save_interval and save_strategy
    args.init_from = None
    # args.out_dir = '/home/pc5090/Code/github/CleanGPT/out/TinyStory/TinyStory_llama_1024_512_12_10/20251023_215406'
    start_timestep = args.out_dir[args.out_dir.rfind('/')+1:]

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
                name = start_timestep,
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