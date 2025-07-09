import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(base_path)

import torch
import torch.distributed as dist
from tqdm import tqdm
from utils.utils_model import load_model
from data.adder.prepare import AdditionDataset
from data.multiplier.prepare import MultiplicationDataset
from data.data import build_dataloader
from model.NanoGPT import NanoGPT
from model.llama import MyLlama

def eval_score_adder(model, tokenizer, dataloader, total, desc=''):
    local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
    world_size = int(os.environ.get("WORLD_SIZE", default='1'))
    raw_model = model.module if hasattr(model, "module") else model
    ndigit = tokenizer.ndigit

    results = []
    with tqdm(total=total, desc=f'[GPU0-{world_size-1}]: {desc}', disable=local_rank!=0) as pbar:
        for batch in dataloader:
            x = batch[0].to(local_rank)     # (batch_size, seql_len)
            if tokenizer.format_vocab is None:
                d1d2 = x[:, :ndigit*2]      # (batch_size, ndigit*2)
            else:
                d1d2 = x[:, :ndigit*2+2]    # (batch_size, ndigit*2+2), +2 for '+' and '='
            
            if isinstance(raw_model, NanoGPT):
                d1d2d3, _ = raw_model.generate(d1d2, ndigit+1, do_sample=False)
            elif isinstance(raw_model, MyLlama):
                d1d2d3, _ = raw_model.generate(d1d2, ndigit+1, do_sample=False)
            else:
                raise ValueError(f"Unsupported model type: {type(raw_model)}")

            correct = tokenizer.decode(d1d2, d1d2d3)    # (batch_size, )
            results.append(correct)

            dist.barrier()
            correct_all = torch.vstack(results).to(torch.float32)
            correct_rate = correct_all.mean().item()
            pbar.set_postfix({'correct_rate': f'{correct_rate:.2f}'})
            pbar.update(len(correct) * world_size)
            
            if correct_all.numel() >= total:
                break

    return correct_rate

def eval_score_multiplier(model, tokenizer, dataloader, total, desc=''):
    local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
    world_size = int(os.environ.get("WORLD_SIZE", default='1'))
    raw_model = model.module if hasattr(model, "module") else model
    ndigit = tokenizer.ndigit

    results = []
    with tqdm(total=total, desc=f'[GPU0-{world_size-1}]: {desc}', disable=local_rank != 0) as pbar:
        for batch in dataloader:
            x = batch[0].to(local_rank)         # (batch_size, seql_len)
            if tokenizer.format_vocab is None:
                d1d2 = x[:, :ndigit*2]          # (batch_size, ndigit*2)
            else:
                d1d2 = x[:, :ndigit*2+2]        # (batch_size, ndigit*2+2), +2 for '+' and '='
            if isinstance(raw_model, NanoGPT):
                d1d2d3, _ = raw_model.generate(d1d2, ndigit*2, do_sample=False)
            elif isinstance(raw_model, MyLlama):
                d1d2d3, _ = raw_model.generate(d1d2, ndigit*2, do_sample=False,)
            else:
                raise ValueError(f"Unsupported model type: {type(raw_model)}")
            
            correct = tokenizer.decode(d1d2, d1d2d3)    # (batch_size, )
            results.append(correct)
            
            if world_size > 1:
                dist.barrier()
            correct_all = torch.vstack(results).to(torch.float32)
            correct_rate = correct_all.mean().item()
            pbar.set_postfix({'correct_rate': f'{correct_rate:.2f}'})
            pbar.update(len(correct) * world_size)

            if correct_all.numel() >= total:
                break

    return correct_rate

def main():
    # load model, tokenizer and decoder
    # out_path = f"{base_path}/out/Multiplier(3_format)_llama_1024_512_8_10"
    out_path = f"{base_path}/out/Multiplier(3_format)_NanoGPT_1024_512_8_10"
    args, model, dataset_name, tokenizer, decoder = load_model(out_path)
    model = model.to('cuda:0').eval()

    # setting
    args.problem_batch_size_per_gpu = 64
    args.problem_batch_num = 4
    
    # generate text & flow printing
    with torch.inference_mode():
        if dataset_name == 'adder':
            problem_dataset = AdditionDataset(tokenizer.ndigit, 'test')
            problem_dataloader = build_dataloader(args, problem_dataset, dataset_type='test')
            total = args.problem_batch_size_per_gpu * args.problem_batch_num
            eval_score_adder(model, tokenizer, problem_dataloader, total, desc=f'Eval on add({tokenizer.ndigit})')
        elif dataset_name == 'multiplier':
            problem_dataset = MultiplicationDataset(tokenizer.ndigit, 'test', format_vocab=tokenizer.format_vocab)
            problem_dataloader = build_dataloader(args, problem_dataset, dataset_type='test')
            total = args.problem_batch_size_per_gpu * args.problem_batch_num
            eval_score_multiplier(model, tokenizer, problem_dataloader, total, desc=f'Eval on mul({tokenizer.ndigit})')

if __name__ == "__main__":
    main()