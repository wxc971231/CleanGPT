
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

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

from utils.utils import create_folder_if_not_exist, create_folder_overwrite_if_exist, set_seed
from data.data import AutoRegressDataset, build_dataloader_AR
from train.config import parse_args
from train.trainer import Trainer
import setproctitle
setproctitle.setproctitle("Template-Test@fff")

# CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=gpu ./train/train_dd[].py 
def ddp_setup():
    os.environ["MASTER_ADDR"] = "localhost" # localhost for single node
    os.environ["MASTER_PORT"] = "21667"     # Any free port
    init_process_group(backend="nccl")      # nccl for linux, gloo for windows
    torch.cuda.set_device(int(os.environ.get("RANK", default='0')))

def get_args_ready(WORLD_SIZE:int, RANK:int):
    ''' train a miniature character-level shakespeare model, good for debugging and playing on macbooks and such '''
    args = parse_args()

    # model setting
    args.model = 'NanoGPT'
    args.n_position = 1024
    args.n_layer = 6
    args.n_head = 6
    args.n_embd = 384
    args.n_inner = 4 * args.n_embd
    args.dropout = 0.0                          # for pretraining 0 is good, for finetuning try 0.1+
    args.init_from = None                       # training from scratch or resuming from latest snapshot within out-dir

    # optimizer setting
    args.lr_begin = 0                                       
    args.lr_max = 1e-3                          # with baby networks can afford to go a bit higher
    args.lr_decay_factor = 10.0                 # min learning rate equals to (learning_rate / 10) usually
    args.lr_warmup_ratio = 0.05
    args.lr_decay_ratio = 0.95
    args.lr_decay_style = "cosine"
    args.wd_begin = 1e-3                        # with baby networks can afford to go a bit higher (1e-4 ~ 1e-2)
    args.wd_end = args.wd_begin                 # For most of situation, keep the weight decay coefficient 'constant' is suitable
    args.wd_decr_style = "constant"            
    args.ga_begin = 2                           # batch_grad_accum is used to simulate larger batch sizes              
    args.ga_end = args.ga_begin                 # with baby networks we can simply use 'constant' grad_accum_step, but for large networks sometimes increase to 2x~10x
    args.grad_accum_step_incr_style = "constant"
    args.adam_beta2 = 0.99                      # make a bit bigger because number of tokens per iter is small

    # training setting
    args.batch_size_per_gpu = 64                                            # training batch_size (per GPU)
    args.batch_size = args.batch_size_per_gpu * WORLD_SIZE * args.ga_begin  # equivalent training batch_size
    args.eval_batch_num = 100
    args.eval_batch_size_per_gpu = 256
    args.eval_batch_size = args.eval_batch_size_per_gpu * WORLD_SIZE
    args.clip_grad = 1.0                        # clip gradients at this value, or disable if == 0.0
    args.early_stopping_patience = 6
    args.early_stopping_delta = 0
    
    # IO setting
    args.dataset = 'shakespeare_char'
    args.wandb_project = 'Template'
    args.exp_name = 'shakespeare-char'
    args.exp_profile = f'{args.exp_name}_{args.n_position}_{args.n_embd}_{args.n_head}_{args.n_layer}'
    args.out_dir = f'{base_path}/out/{args.exp_name}'
    args.save_strategy = 'best'                 # 'best' or 'interval'

    # other setting, which are usually changed
    args.seeds = [42, ]                         # random seeds
    args.weight_tie = True                      # tie the word embedding and softmax weights, like in GPT-2
    args.add_bias = False                       # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    args.override_opt_param_scheduler = False   # Set 'True' to override all scheduler setting, otherwise the scheduler will be set by checkpoint
    args.wandb = False                          # use wandb to log training info
    args.use_early_stopping = True              # use the early stopping mechanism to aviod overfitting
    args.save_ckpt = True                       # update ckpt by save_interval and save_strategy
    args.save_snapshot = True                   # save the latest traing snapshot, from which we can resume training 
    args.use_kvcache = True                     # use kv cache to speed up evaluation          
    args.use_amp = False                        # use automatic mixed precision (AMP) to speed up training, which may hurt the performance
    args.compile = torch.__version__ >= "2.0"   # use PyTorch 2.0 to compile the model to be faster
    args.train_iters = 5000                     # total batch_num
    args.eval_interval = 100                    # keep frequent because we'll overfit
    args.save_interval = 100   

    # assert some hyper paras
    assert args.train_iters % args.batch_grad_accum_step == 0
    assert args.train_iters % args.eval_interval == 0
    assert args.train_iters % args.save_interval == 0

    # get ready for wandb logging
    if RANK == 0:
        create_folder_if_not_exist(f'{base_path}/Wandb')
    if not args.wandb:       
        os.environ['WANDB_MODE'] = 'offline'

    # create floder to save ckpts and hyperparas if we need
    if (args.save_ckpt or args.save_snapshot) and RANK == 0:
        create_folder_if_not_exist(f'{args.out_dir}/{args.save_strategy}')
        with open(f'{args.out_dir}/config.json', 'w') as f:
            f.write(json.dumps(vars(args), indent=4))
        shutil.copy2(
            src=os.path.abspath(__file__),
            dst=f'{args.out_dir}/train_ddp.py',
        )

    return args

def load_dataset(args):
    # build dataset
    dataset_train = AutoRegressDataset(args, f'{base_path}/data/{args.dataset}/train.bin')
    dataset_val = AutoRegressDataset(args, f'{base_path}/data/{args.dataset}/val.bin')
    dataset_test = AutoRegressDataset(args, f'{base_path}/data/{args.dataset}/test.bin')
    dataset_dict = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}

    # update args.vocab_size
    meta_path = os.path.join(f'{base_path}/data/{args.dataset}/meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    args.vocab_size = meta['vocab_size']
    print(f"found vocab_size = {args.vocab_size} (inside {meta_path})")

    return dataset_dict


if __name__ == "__main__":
    # init DDP process group
    #ddp_setup()
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", default='1'))
    RANK = int(os.environ.get("RANK", default='0'))

    # activate tf32 on matmul and cudnn to boost NVIDIA Ampere GPU performance
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    
    # get hyper paras ready
    args = get_args_ready(WORLD_SIZE, RANK)

    # build training objs
    dataset_dict = load_dataset(args)

    # train
    for seed in args.seeds:
        if args.save_ckpt and args.save_strategy == 'interval' and RANK == 0:
            create_folder_overwrite_if_exist(f'{args.out_dir}/interval/{seed}')

        # This unique id is necessary for log resuming
        wandb_id = wandb.util.generate_id() 
        
        # build trainer
        trianer = Trainer(args, seed, wandb_id, dataset_dict)

        # wandb log only on rank0
        if RANK == 0:
            with wandb.init(
                project=args.wandb_project,
                group = args.exp_name,
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