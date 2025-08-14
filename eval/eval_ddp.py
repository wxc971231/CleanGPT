import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import pickle
from torch.distributed import init_process_group, destroy_process_group
from eval.config import parse_args
from utils.utils_model import load_model
from eval.evaluater import Evaluater
from data.data import AutoRegressDataset, AdditionDataset, AdditionTokenizer, MultiplicationDataset, MultiplicationTokenizer
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

def get_args_ready(WORLD_SIZE:int, RANK:int):
    ''' train a miniature character-level shakespeare model, good for debugging and playing on macbooks and such '''
    eval_args = parse_args()
    eval_args.seed = 42                      # random seeds
    eval_args.num_workers = 0                # dataloader workers
    
    # model setting
    # eval_args.out_path = f'{base_path}/out/TinyStory_NanoGPT_1024_512_8_10'
    # eval_args.out_path = f'{base_path}/out/Multiplier(3_format)_llama_1024_512_8_10'          # dataset name
    # eval_args.out_path = f'{base_path}/out/Multiplier(3_format)_NanoGPT_1024_768_12_10'
    eval_args.out_path = f'{base_path}/out/Adder(3_format)/Adder(3_format)_NanoGPT_1024_256_8_4/20250814_101733'
    
    # training setting
    eval_args.batch_size_per_gpu = 256                                           # training batch_size (per GPU)
    eval_args.batch_size = eval_args.batch_size_per_gpu * WORLD_SIZE
    eval_args.batch_num = 64
    eval_args.eval_batch_num = 20
    eval_args.eval_batch_size_per_gpu = 64
    eval_args.eval_batch_size = eval_args.eval_batch_size_per_gpu * WORLD_SIZE
    eval_args.problem_batch_num = 2
    eval_args.problem_batch_size_per_gpu = 128      # eval problem batch_size (per GPU)
    eval_args.total_problem_num = eval_args.problem_batch_size_per_gpu * WORLD_SIZE * eval_args.problem_batch_num
    eval_args.resample_times = 8
    eval_args.check_loss = True

    return eval_args

def get_eval_components(eval_args, RANK):
    # load model
    args, model, dataset_name, tokenizer, decoder = load_model(eval_args.out_path)
    model = model.to(RANK).eval()
    
    args.batch_size_per_gpu = eval_args.batch_size_per_gpu
    args.batch_size = eval_args.batch_size
    args.batch_num = eval_args.batch_num
    args.eval_batch_num = eval_args.eval_batch_num
    args.eval_batch_size_per_gpu = eval_args.eval_batch_size_per_gpu
    args.eval_batch_size = eval_args.eval_batch_size
    args.problem_batch_num = eval_args.problem_batch_num
    args.problem_batch_size_per_gpu = eval_args.problem_batch_size_per_gpu
    args.total_problem_num = eval_args.total_problem_num
    args.resample_times = eval_args.resample_times
    args.vocab_size = tokenizer.vocab_size

    # load dataset
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
    return args, model, dataset_name, dataset_dict, tokenizer

if __name__ == "__main__":
    # init DDP process group
    ddp_setup()
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", default='1'))
    RANK = int(os.environ.get("RANK", default='0'))

    # activate tf32 on matmul and cudnn to boost NVIDIA Ampere GPU performance
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    
    # get hyper paras ready
    eval_args = get_args_ready(WORLD_SIZE, RANK)

    # build training objs
    args, model, dataset_name, dataset_dict, tokenizer = get_eval_components(eval_args, RANK)

    # build evaluater
    evaluater = Evaluater(args, eval_args, model, dataset_dict, tokenizer)

    # evaluate
    evaluater.evaluate()

    # destroy DDP process group
    destroy_process_group()