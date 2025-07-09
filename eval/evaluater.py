import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import os
import torch
import wandb
import time
import glob
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Any, Dict
from calflops import calculate_flops

from data.data import build_dataloader
from train.scheduler import EarlyStopping, OptimizerParamScheduler
from model.llama import MyLlama, MyLlamaConfig
from model.NanoGPT import NanoGPT, NanoGPTConfig
from utils.utils_model import remove_compiled_prefix
from utils.utils import set_seed, clean_print
from eval.script_score import eval_score_adder, eval_score_multiplier

class Evaluater:
    def __init__(self, args, eval_args, model, dataset_dict, tokenizer=None):
        self.args = args
        self.eval_args = eval_args
        self.dataset_dict = dataset_dict
        self.tokenizer = tokenizer
        self.world_size = int(os.environ.get("WORLD_SIZE", default='1'))
        self.local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
        self.batch_now = {'train': 0, 'val': 0, 'test': 0}
        set_seed(eval_args.seed)

        # data stuff
        self.dataloader_dict = self._prepare_dataloader(dataset_dict)       

        # AMP setting
        if args.use_amp:        
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
            self.scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16)) # bfloat16 doesn't need GradScaler cause it can be calculated in hardware drictly   
        else:
            self.ctx = nullcontext()                                                    # null context
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)                      # no-op

        # eval setting
        self.eval_setting = {
            'greedy': lambda: self._eval_score(sample=False),
            # 'sample': lambda: self._eval_score(sample=True),
        }

        # model
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model

    def _prepare_dataloader(self, dataset_dict:dict):
        dataloader_dict = {}
        for data_type, dataset in dataset_dict.items():            
            dataloader = None if dataset is None else \
                build_dataloader(
                    self.args, dataset, data_type, 
                    current_batch=self.batch_now[data_type], seed=self.eval_args.seed
                )
            dataloader_dict[data_type] = dataloader

        train_batch_size_begin = self.args.batch_size_per_gpu * self.world_size
        train_batch_size_end = self.args.batch_size_per_gpu * self.world_size
        train_sample_all = self.args.batch_size_per_gpu * self.world_size * self.args.train_iters
        if self.local_rank == 0:
            print('\n' + '-'*20 + 'Data Info' + '-'*20)
            print(f"> vocab_size:             {self.args.vocab_size}")
            print(f"> Train Dataset Size:     {len(dataset_dict['train'])}")
            print(f"> Train sample all:       {train_sample_all}")
            print(f"> Train batch_size:       {train_batch_size_begin} ({self.args.batch_size_per_gpu}*{self.world_size}*{self.args.ga_begin}) ---{self.args.grad_accum_step_incr_style}---> {train_batch_size_end} ({self.args.batch_size_per_gpu}*{self.world_size}*{self.args.ga_end})")
            print(f"> Val Dataset size:       {len(dataset_dict['val'])}")
            print(f"> Val/Eval batch_size:    {self.args.eval_batch_size} ({self.args.eval_batch_size_per_gpu}*{self.world_size})")
            print('-'*50 + '\n')
        return dataloader_dict

    def _run_batch(self, data) -> float:
        with self.ctx:
            _, loss = self.model(*tuple(data))
        
        return loss.item()

    def _run_macro_batch(self, is_train=False):
        macro_batch_now = self.batch_now['train'] // self.args.batch_num
        if is_train:
            total = self.args.batch_num
            desc = f"[GPU0-{self.world_size-1}]: Trianing batch {macro_batch_now}({self.batch_now['train']}-{self.batch_now['train'] + self.args.batch_num})"
            dataloader = self.dataloader_dict['train']
            dataloader.sampler.set_batch(self.batch_now['train'], macro_batch_now, True)     
        else:
            total = self.args.eval_batch_num
            desc = f'[GPU0-{self.world_size-1}]: Calculating val_loss'
            dataloader = self.dataloader_dict['val']
            dataloader.sampler.set_batch(self.batch_now['val'], macro_batch_now, False)     
   
        losses = []
        with tqdm(total=total, desc=desc, position=self.local_rank, disable=self.local_rank!=0) as pbar:
            for i, batch in enumerate(dataloader):
                batch = [x.to(self.local_rank) for x in batch]
            
                # run batch
                loss = self._run_batch(batch[:-1])
                losses.append(loss)

                # update progress bar
                batch_token = self.args.batch_size * self.args.n_position if is_train else \
                              self.args.eval_batch_size * self.args.n_position
                pbar.set_postfix({
                    'batch_token': f'{batch_token/1e6:.4f}M',
                    'loss':f'{loss:.2f}', 
                    'ave loss (latest 20)': f'{np.array(losses[-20:]).mean():.2f}',
                })
                pbar.update()
                
                if i == total - 1:
                    break
        
        batches_loss = np.array(losses).mean()
        return batches_loss

    def _eval_score(self, sample=False):
        desc_sample = f'sample{self.args.resample_times}' if sample else 'greedy'
        desc = f'Evaluating on {self.args.dataset}({desc_sample})'
        problem_dataloader = self.dataloader_dict['test']
        total = self.args.total_problem_num

        if self.args.dataset == 'adder':
            assert sample == False, "Sample mode is not supported for adder dataset"
            correct_rate = eval_score_adder(self.model, self.tokenizer, problem_dataloader, total, desc)
            return correct_rate
        elif self.args.dataset == 'multiplier':
            assert sample == False, "Sample mode is not supported for multiplier dataset"
            correct_rate = eval_score_multiplier(self.model, self.tokenizer, problem_dataloader, total, desc)
            return correct_rate
        elif self.args.dataset in ['tinystory', 'shakespeare_char']:
            return 0
        else:
            raise Exception(f"Unknown dataset: {self.args.dataset}")

    def evaluate(self):    
        torch.cuda.empty_cache()
        log_dict = {}

        # Check loss
        if self.eval_args.check_loss:
            self.model.eval()
            with torch.inference_mode():
                train_loss = self._run_macro_batch(is_train=True)
                val_loss = self._run_macro_batch(is_train=False)
                log_dict.update({'train_loss': train_loss, 'val_loss': val_loss})

        # Evaluate policy performance at specified batch intervals
        self.model.eval()
        with torch.inference_mode():  
            for setting, eval_func in self.eval_setting.items():    
                score = eval_func()        
                log_dict.update({f'{self.args.dataset}/score({setting})': score})
                
        # gather trainig info from all GPU
        log_tensor = torch.tensor([log_dict[k] for k in sorted(log_dict)]).to(self.local_rank)
        log_gather_list = [torch.zeros_like(log_tensor) for _ in range(self.world_size)]
        dist.gather(log_tensor, log_gather_list if self.local_rank == 0 else None, dst=0)
        
        # only do file saving and logging at rank0
        if self.local_rank == 0:
            # update trainig info
            log_value_tensor = torch.mean(torch.stack(log_gather_list), axis=0, dtype=torch.float32)
            for i, key in enumerate(sorted(log_dict)):
                log_dict[key] = log_value_tensor[i].item()

            import pprint
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint({k: round(v,5) for k,v in log_dict.items()})