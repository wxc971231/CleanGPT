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

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    total_steps: int            # for scheduler initialization
    trained_time: float         # for total training time
    latest_batch: Dict[int, Any]# for consistent optimizer-para schedule
    best_val_loss: float        # for early_stopping
    wandb_id: str               # for resuming wandb log

class Trainer:
    def __init__(self, args, seed, wandb_id, dataset_dict, tokenizer=None):
        self.args = args
        self.seed = seed
        self.wandb_id = wandb_id
        self.dataset_dict = dataset_dict
        self.tokenizer = tokenizer
        self.snapshot_path = f'{self.args.out_dir}/snapshot_seed{seed}.pt'
        self.world_size = int(os.environ.get("WORLD_SIZE", default='1'))
        self.local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
        set_seed(seed)   

        self.ga_current = self.args.ga_begin
        self.best_val_loss = float('inf')
        self.batch_start = 0
        self.batch_now = {'train': 0, 'val': 0, 'test': 0}
        self.trained_time = 0
        self.grad_batch_cnt = 1

        # build training components
        self._build_model()
        self.early_stopping = EarlyStopping(patience=args.early_stopping_patience, delta=args.early_stopping_delta)
        self.optimizer = self.raw_model.configure_optimizers(self.args, self.local_rank) if args.model == 'NanoGPT' else \
                         self.raw_model.configure_optimizers()
        self.scheduler = OptimizerParamScheduler(self.args, self.optimizer)    
        self._init_model()

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
        self.eval_setting = {}
        if self.args.dataset in ['adder', 'multiplier']:
            self.eval_setting = {
                'greedy': lambda: self._eval_score(sample=False),
                # 'sample': lambda: self._eval_score(sample=True),
            }

    def _build_model(self):
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
        if self.args.model == 'NanoGPT':
            model_args = {key: getattr(self.args, key) for key in [
                'n_position', 'n_layer', 'n_head', 'n_embed', 'vocab_size', 'dropout', 'dropout_attn', 'add_bias', 'weight_tying'
            ]}
            model_args.update({'mask_out_token': pad_token_id})
            gpt_conf = NanoGPTConfig(**model_args)
            self.raw_model = NanoGPT(gpt_conf).to(self.local_rank).train()
        elif self.args.model == 'llama':
            model_args = {key: getattr(self.args, key) for key in [
                'n_position', 'n_layer', 'n_q_head', 'n_kv_head', 'n_embed', 'vocab_size', 
                'rms_norm_eps', 'weight_tying', 'dropout_attn', 'add_bias', 'lr_begin',
                'adam_beta1', 'adam_beta2', 'adam_eps', 
            ]}
            model_args.update({
                'pad_token_id': pad_token_id,
                'mask_out_token': pad_token_id,
                'weight_decay': self.args.wd_begin,
            })
            assert self.args.wd_decr_style == "constant"
            llama_conf = MyLlamaConfig(**model_args)
            self.raw_model = MyLlama(llama_conf).to(self.local_rank).train()
        else:
            raise Exception(f'{self.args.model} is not support currently')
    
        # Wrap the model DDP, which synch model across all the processes.
        self.model = DDP(self.raw_model, device_ids=[self.local_rank])
        self.raw_model = self.model.module

    def _init_model(self):
        # init from latest snapshot or from scratch
        if self.args.init_from is None:
            try:
                # try to resume training from the latest snapshot
                snapshot_path = f'{self.args.out_dir}/snapshot_seed{self.seed}.pt'
                snapshot_data = torch.load(snapshot_path, map_location=f"cuda:{self.local_rank}")
                snapshot = Snapshot(**snapshot_data)
                self.raw_model.load_state_dict(remove_compiled_prefix(snapshot.model_state))
                self.optimizer.load_state_dict(snapshot.optimizer_state)
                batch_factor = self.scheduler.load_state_dict(snapshot.scheduler_state)
                self.ga_current = self.scheduler.ga_scheduler.step(0)
                self.best_val_loss = snapshot.best_val_loss
                self.batch_start = self.batch_now = {k: int(v * batch_factor) for k, v in snapshot.latest_batch.items()}
                self.wandb_id = snapshot.wandb_id
                self.trained_time = snapshot.trained_time
                macro_batch_now = self.batch_now['train'] // self.args.batch_num
                clean_print(f"Resuming training from snapshot at macro_batch {macro_batch_now}({self.batch_now['train']}) with grad_accum_step [{self.ga_current}]", self.local_rank, '[Trainer]')

            except FileNotFoundError:    
                # if no snapshot found, start from scratch
                self.best_val_loss = float('inf')
                self.batch_start = self.batch_now = {'train': 0, 'val': 0, 'test': 0}
                self.ga_current = self.args.ga_begin
                self.trained_time = 0
                clean_print(f"Snapshot not found. Training model from scratch with grad_accum_step={self.ga_current}", self.local_rank, '[Trainer]')
            
        # init from pretrained_ckpt to finetune
        elif self.args.init_from.endswith('.pt'):
            ckpt_model_state = torch.load(self.args.init_from, map_location=f"cuda:{self.local_rank}")
            self.raw_model.load_state_dict(remove_compiled_prefix(ckpt_model_state))
            clean_print(f"Pretrained model [{self.args.init_from}] loaded. Fine-tuning from scratch with grad_accum_step [{self.ga_current}]", self.local_rank, '[Trainer]')

        # special for NanoGPT, init from huggingface GPT2 ckpt
        elif self.args.model == 'NanoGPT' and self.args.init_from.startswith('gpt2'):
            override_args = dict(dropout=self.args.dropout) # only dropout can be overwritten
            self.raw_model = self.raw_model.from_pretrained(self.args.init_from, override_args)
            # override the created config params, so we can store them into checkpoint correctly
            for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
                setattr(self.args, getattr(self.raw_model.config, k))
            clean_print(f"Resuming training from OpenAI GPT-2 weights: [{self.args.init_from}]", self.local_rank, '[Trainer]')
        else:
            raise Exception(f"The model can conly be init from a .pt file path or a gpt2-model name (when model=NanoGPT), instead of {self.args.init_from}")

        # crop down the model block size if desired, using model surgery
        if self.args.model == 'NanoGPT' and self.args.n_position < self.raw_model.config.n_position:
            self.raw_model.crop_block_size(self.args.n_position)
            clean_print(f'The input-seq-length has been crop to [{self.args.n_position}] as args setting', self.local_rank, '[Trainer]')
        
        # check MACs and params quantity, which can only work befor torhc.compile
        flops, macs, params = self._check_MACs()

        # reset weight tying in case the model is loaded from a snapshot or pretrained ckpt
        if self.args.weight_tying:
            if self.args.model == 'NanoGPT':
                self.raw_model.transformer.wte.weight = self.raw_model.lm_head.weight
            elif self.args.model == 'llama':
                self.raw_model.lm_head.weight = self.raw_model.model.embed_tokens.weight
           
        # compile the model
        if self.args.compile:
            clean_print("compiling the model... (takes a ~minute)", self.local_rank, '[Trainer]')
            self.raw_model = torch.compile(self.raw_model)      # requires PyTorch 2.0

        if self.local_rank == 0:
            print('\n' + '-'*20 + 'Model Info' + '-'*20)
            print(f'> Model Type:           {self.args.model}')
            print(f"> Total model params:   {params/1e6:.2f} M")
            print(f'> MACs per val batch:   {macs*self.world_size/1e9:.2f} G')
            print(f'> Flops per val batch:  {flops*self.world_size/1e9:.2f} G')
            print(f'> Using kv-cache:       {self.args.use_kvcache}')
            print(f'> Using torch.compile:  {self.args.compile}')
            print('-'*50 + '\n')
        
    def _prepare_dataloader(self, dataset_dict:dict):
        dataloader_dict = {}
        for data_type, dataset in dataset_dict.items():            
            dataloader = None if dataset is None else \
                build_dataloader(
                    self.args, dataset, data_type, 
                    current_batch=self.batch_now[data_type], seed=self.seed
                )
            dataloader_dict[data_type] = dataloader

        train_batch_size_begin = self.args.batch_size_per_gpu * self.world_size * self.args.ga_begin
        train_batch_size_end = self.args.batch_size_per_gpu * self.world_size * self.args.ga_end
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

    def _check_MACs(self):
        def _get_dummy_data():
            dataloader = build_dataloader(self.args, self.dataset_dict['val'], dataset_type='val')
            dummy_data = next(dataloader.__iter__())
            dummy_data = [x.to(self.local_rank) for x in dummy_data]
            data, _ = dummy_data[:-1], dummy_data[-1]
            del dataloader
            return data
        
        flops, macs, params = None, None, None
        with torch.inference_mode():
            if self.local_rank == 0:
                dummy_data = _get_dummy_data()
                flops, macs, params = calculate_flops(
                    model=self.raw_model,
                    args=dummy_data,
                    print_results=False,
                    output_as_string=False
                )
        dist.barrier()
        return flops, macs, params

    def _save_snapshot(self):
        snapshot = Snapshot(
            model_state=self.raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            total_steps=self.args.train_iters,
            trained_time=self.trained_time,
            latest_batch=self.batch_now.copy(),
            best_val_loss=self.best_val_loss,
            wandb_id=self.wandb_id
        )
        torch.save(asdict(snapshot), self.snapshot_path)

    def _save_checkpoint(self, val_loss=float('inf')):
        ckpt_dir = None
        macro_batch_now = self.batch_now['train'] // self.args.batch_num
        if self.args.save_strategy == 'interval':
            torch.save(
                self.raw_model.state_dict(),
                f"{self.args.out_dir}/interval/{self.seed}/{round(val_loss,3)}_seed{self.seed}_mb{macro_batch_now}.pt"
            )
            ckpt_dir = os.path.join(f'{self.args.out_dir}/interval/{self.seed}', '*.pt')
        elif self.args.save_strategy == 'best':
            if val_loss < self.best_val_loss: 
                self.best_val_loss = val_loss         
                torch.save(
                    self.raw_model.state_dict(),
                    f"{self.args.out_dir}/best/{round(self.best_val_loss,3)}_seed{self.seed}_mb{macro_batch_now}.pt"
                )
                ckpt_dir = os.path.join(f'{self.args.out_dir}/best', '*.pt')
        else:
            raise Exception(f'Unknown save strategy: {self.args.save_strategy}')
        
        # Only keep the best n ckpt
        if ckpt_dir:
            ckpt_files = glob.glob(ckpt_dir)
            ckpt_files = sorted(ckpt_files, key=lambda x: float(os.path.basename(x).split('_')[0]))
            for ckpt in ckpt_files[self.args.save_ckpt_num:]:
                os.remove(ckpt)

    def _run_batch(self, data, is_train=False) -> float:
        with self.ctx:
            _, loss = self.model(*tuple(data))
        
        new_lr = new_wd = grad_norm = None
        if is_train:
            loss_ave = loss / self.ga_current
            if self.grad_batch_cnt % self.ga_current != 0:
                # In DDP training we only need to sync gradients at the last micro step.
                with self.model.no_sync():
                    self.scaler.scale(loss_ave).backward()
            else:
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss_ave).backward()

                # clip the gradient
                if self.args.clip_grad != 0:
                    # When using AMP, the gradients are stored in scaled form, We need to unscale the gradients to perform the correct clipping operation.
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                
                # When using AMP, scaler can adjust gradients and skip update step if necessary to avoid numerical instability.
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # flush the gradients as soon as we can, no need for this memory anymore
                self.optimizer.zero_grad(set_to_none=True)
                
                # update lr, grad_accum_step and weight_decay
                new_lr, new_ga_step, new_wd = self.scheduler.step(self.ga_current)
                self.ga_current = new_ga_step
                self.grad_batch_cnt = 0
            
            self.grad_batch_cnt += 1

        return loss.item(), grad_norm, new_lr, new_wd

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

        self.optimizer.zero_grad()        
        losses, visited_data = [], []
        new_lr = new_wd = grad_norm = 0
        with tqdm(total=total, desc=desc, position=self.local_rank, disable=self.local_rank!=0) as pbar:
            for i, batch in enumerate(dataloader):
                batch = [x.to(self.local_rank) for x in batch]
                data, dataset_idxs = batch[:-1], batch[-1]
                if is_train:
                    visited_data.append(dataset_idxs % len(self.dataset_dict['train']))

                # run batch
                loss, norm, lr, wd = self._run_batch(data, is_train)
                new_lr = lr if lr is not None else new_lr
                new_wd = wd if wd is not None else new_wd
                grad_norm = norm.item() if norm is not None else grad_norm
                losses.append(loss)

                # update progress bar
                batch_token = self.args.batch_size_per_gpu * self.world_size * self.ga_current * self.args.n_position if is_train else \
                                self.args.eval_batch_size * self.args.n_position
                pbar.set_postfix({
                    'grad_norm': f'{grad_norm:.4f}',
                    'batch_token': f'{batch_token/1e6:.4f}M',
                    'loss':f'{loss:.2f}', 
                    'ave loss (latest 20)': f'{np.array(losses[-20:]).mean():.2f}',
                })
                pbar.update()
                
                if i == total - 1:
                    break
        
        if is_train: self.batch_now['train'] += total
        else: self.batch_now['val'] += total

        batches_loss = np.array(losses).mean()
        return batches_loss, new_lr, new_wd, grad_norm, batch_token, visited_data

    def _eval_score(self, sample=False):
        desc_sample = f'sample{self.args.resample_times}' if sample else 'greedy'
        desc = f'Evaluating on {self.args.dataset}({desc_sample})'
        problem_dataloader = self.dataloader_dict['test']
        total = self.args.total_problem_num
        
        if self.args.dataset == 'adder':
            assert sample == False, "Sample mode is not supported for adder dataset"
            correct_rate = eval_score_adder(self.raw_model, self.tokenizer, problem_dataloader, total, desc)
            return correct_rate
        elif self.args.dataset == 'multiplier':
            assert sample == False, "Sample mode is not supported for multiplier dataset"
            correct_rate = eval_score_multiplier(self.raw_model, self.tokenizer, problem_dataloader, total, desc)
            return correct_rate
        else:
            raise Exception(f"Unknown dataset: {self.args.dataset}")

    def train(self):    
        early_stop_signal = torch.tensor([0,]).to(self.local_rank)
        if self.local_rank == 0:
            dataset_visited_cnt = np.zeros(len(self.dataset_dict['train']), dtype=np.int8)

        # start training
        for batch in range(self.batch_start['train'], self.args.train_iters + 1, self.args.batch_num):  
            self.batch_now['train'] = batch
            wandb_log_dict = {}

            # Save snapshot before the batch training process, avoid overlap when resuming from the snapshot
            if self.args.save_snapshot and batch % self.args.save_interval == 0 and batch != 0 and self.local_rank == 0 :
                self._save_snapshot()
            
            # Calculate validation losses at specified batch intervals
            if (batch % self.args.eval_interval == 0 or batch == self.args.train_iters) and (batch != 0 or not self.args.skip_first_eval):
                self.model.eval()
                with torch.no_grad():
                    val_loss, _, _, _, _, _ = self._run_macro_batch(is_train=False)
                    wandb_log_dict.update({f"{self.args.dataset}/val_loss": val_loss})                    

            # Evaluate policy performance at specified batch intervals
            if (batch % self.args.eval_score_interval == 0 or batch == self.args.train_iters) and (batch != 0 or not self.args.skip_first_eval):
                self.model.eval()
                with torch.inference_mode():  
                    for setting, eval_func in self.eval_setting.items():
                        if self.args.dataset == 'adder':
                            correct_rate = eval_func()
                            wandb_log_dict.update({f'{self.args.dataset}/correct_rate({setting})': correct_rate})
                        elif self.args.dataset == 'multiplier':
                            correct_rate = eval_func()
                            wandb_log_dict.update({f'{self.args.dataset}/correct_rate({setting})': correct_rate})
                        else:
                            raise Exception(f"Unknown dataset: {self.args.dataset}")

            # exit point
            if batch == self.args.train_iters:
                break
            
            # clear CUDA cache to avoid OOM
            torch.cuda.empty_cache()

            # training process
            self.model.train()
            start_time = time.time()
            train_loss, new_lr, new_wd, grad_norm, batch_token, visited_data = self._run_macro_batch(is_train=True)
            self.trained_time += time.time() - start_time

            wandb_log_dict.update({
                f"{self.args.dataset}/batch": batch,
                f"{self.args.dataset}/m_batch": int(batch/self.args.batch_num),
                f'{self.args.dataset}/lr': new_lr,
                f'{self.args.dataset}/wd': new_wd,
                f'{self.args.dataset}/grad_norm': grad_norm,
                f'{self.args.dataset}/batch_token': batch_token,
                f'{self.args.dataset}/train_loss': train_loss, 
                f'{self.args.dataset}/trained_time': self.trained_time,
            })    

            # gather trainig info from all GPU
            log_tensor = torch.tensor([wandb_log_dict[k] for k in sorted(wandb_log_dict)]).to(self.local_rank)
            log_gather_list = [torch.zeros_like(log_tensor) for _ in range(self.world_size)]
            dist.gather(log_tensor, log_gather_list if self.local_rank == 0 else None, dst=0)
                
            # gather dataset visited info from all GPU
            visited_tensor = torch.stack(visited_data).flatten()
            visited_gather_list = [torch.zeros_like(visited_tensor) for _ in range(self.world_size)]
            dist.gather(visited_tensor, visited_gather_list if self.local_rank == 0 else None, dst = 0)

            # only do file saving and logging at rank0
            if self.local_rank == 0:
                # update data-counter
                batches_visited = torch.cat(visited_gather_list).cpu()
                dataset_visited_cnt[batches_visited] += 1

                # update trainig info
                log_value_tensor = torch.mean(torch.stack(log_gather_list), axis=0, dtype=torch.float32)
                for i, key in enumerate(sorted(wandb_log_dict)):
                    wandb_log_dict[key] = log_value_tensor[i].item()
                
                    # print training info
                    if self.args.dataset == 'adder' and 'correct_rate' in key:
                        clean_print(f"Adder {key} = {wandb_log_dict[key]:.4f}", self.local_rank, '[Trainer]')
                    elif self.args.dataset == 'multiplier' and 'correct_rate' in key:
                        clean_print(f"Multiplier {key} = {wandb_log_dict[key]:.4f}", self.local_rank, '[Trainer]')

                    # early-stopping and save-ckpt by val_loss 
                    if key == f'{self.args.dataset}/val_loss':
                        val_loss = wandb_log_dict[key]
                        self.early_stopping(val_loss)
                        if self.args.save_ckpt and ((self.args.save_strategy == 'best' and batch != 0) or (self.args.save_strategy == 'interval' and batch % self.args.save_interval == 0)):
                            self._save_checkpoint(val_loss=val_loss)

                # update dataset visited info
                visited_part = dataset_visited_cnt[dataset_visited_cnt!=0]
                wandb_log_dict.update({f'{self.args.dataset}/visited_ratio': len(visited_part)/len(dataset_visited_cnt)})   
                wandb.run.summary[f"{self.args.dataset}/visited_times"] = np.mean(visited_part)
                wandb.log(wandb_log_dict)
                #print(len(visited_part)/len(dataset_visited_cnt), np.mean(visited_part))

                # early stopping
                if self.args.use_early_stopping and self.early_stopping.early_stop:
                    early_stop_signal = torch.tensor([1,]).to(self.local_rank)

            # early stopping exit point    
            dist.broadcast(tensor=early_stop_signal, src=0)
            if early_stop_signal:
                print(f'[GPU{self.local_rank}] Early Stop')
                break