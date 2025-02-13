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

from utils.utils import set_seed, clean_print, remove_compiled_prefix
from data.data import build_dataloader_AR
from train.scheduler import EarlyStopping, OptimizerParamScheduler
from model.NanoGPT import NanoGPT, NanoGPTConfig

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    total_steps: int         # for scheduler initialization
    trained_time: float     # for total training time
    latest_train_batch: int # for consistent optimizer-para schedule
    latest_val_batch: int   # for consistent visit val_dataset
    best_val_loss: float    # for early_stopping
    wandb_id: str           # for resuming wandb log

class Trainer:
    def __init__(self, args, seed, wandb_id, dataset_dict:dict):
        self.args = args
        self.seed = seed
        self.wandb_id = wandb_id
        self.dataset_dict = dataset_dict
        self.snapshot_path = f'{self.args.out_dir}/snapshot_seed{seed}.pt'
        self.world_size = int(os.environ.get("WORLD_SIZE", default='1'))
        self.local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
        set_seed(seed)   

        self.ga_current = self.args.ga_begin
        self.best_val_loss = float('inf')
        self.batch_start = 0
        self.train_batch_now = 0
        self.val_batch_now = 0
        self.trained_time = 0
        self.grad_batch_cnt = 1

        # build training components
        self._build_model()
        self.early_stopping = EarlyStopping(patience=args.early_stopping_patience, delta=args.early_stopping_delta)
        self.optimizer = self.raw_model.configure_optimizers(self.args, self.local_rank)  
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

    def _build_model(self):
        if self.args.model == 'NanoGPT':
            model_args = {key: getattr(self.args, key) for key in [
                'n_position', 'n_layer', 'n_head', 'n_embd', 'vocab_size', 'dropout', 'add_bias', 'weight_tying'
            ]}
            gptconf = NanoGPTConfig(**model_args)
            self.raw_model = NanoGPT(gptconf).to(self.local_rank).train()
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
                self.scheduler.load_state_dict(snapshot.scheduler_state)
                self.ga_current = self.scheduler.get_ga_step()
                self.best_val_loss = snapshot.best_val_loss
                self.batch_start = snapshot.latest_train_batch  
                self.train_batch_now = snapshot.latest_train_batch
                self.val_batch_now = snapshot.latest_val_batch
                self.wandb_id = snapshot.wandb_id
                self.trained_time = snapshot.trained_time
                clean_print(f"Resuming training from snapshot at Batch [{self.train_batch_now}] with grad_accum_step [{self.ga_current}]", self.local_rank, '[Trainer]')

            except FileNotFoundError:    
                # if no snapshot found, start from scratch
                self.best_val_loss = float('inf')
                self.batch_start = 0
                self.train_batch_now = 0
                self.val_batch_now = 0
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
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
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

        # compile the model
        if self.args.compile:
            clean_print("compiling the model... (takes a ~minute)", self.local_rank, '[Trainer]')
            self.raw_model = torch.compile(self.raw_model)      # requires PyTorch 2.0

        if self.local_rank == 0:
            print('\n' + '-'*20 + 'Model Info' + '-'*20)
            print(f'> Model Type:           {self.args.model}')
            print(f"> Total model params:   {params/1e6:.2f} M")
            print(f'> MACs per eval batch:  {macs*self.world_size/1e9:.2f} G')
            print(f'> Flops per eval batch: {flops*self.world_size/1e9:.2f} G')
            print(f'> Using kv-cache:       {self.args.use_kvcache}')
            print(f'> Using torch.compile:  {self.args.compile}')
            print('-'*50 + '\n')
        
    def _prepare_dataloader(self, dataset_dict:dict):
        dataloader_dict = {}
        current_batch_dict = {'train': self.train_batch_now, 'val': self.val_batch_now, 'test': 0}
        for data_type, dataset in dataset_dict.items():
            dataloader = build_dataloader_AR(
                self.args, dataset, is_eval=(data_type!='train'), 
                current_batch=current_batch_dict[data_type], seed=self.seed
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
            print(f"> Test Dataset Size:      {len(dataset_dict['test'])}")
            print(f"> Val/Eval batch_size:    {self.args.eval_batch_size} ({self.args.eval_batch_size_per_gpu}*{self.world_size})")
            print('-'*50 + '\n')
        return dataloader_dict

    def _check_MACs(self):
        def _get_dummy_data():
            dataloader = build_dataloader_AR(self.args, self.dataset_dict['test'], is_eval=True)
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
            latest_train_batch=self.train_batch_now,
            latest_val_batch=self.val_batch_now,
            best_val_loss=self.best_val_loss,
            wandb_id=self.wandb_id
        )
        torch.save(asdict(snapshot), self.snapshot_path)

    def _save_checkpoint(self, val_loss=float('inf')):
        ckpt_dir = None
        if self.args.save_strategy == 'interval':
            torch.save(
                self.raw_model.state_dict(),
                f'{self.args.out_dir}/interval/{self.seed}/{round(val_loss,3)}_seed{self.seed}_batch{self.train_batch_now}.pt'
            )
            ckpt_dir = os.path.join(f'{self.args.out_dir}/interval/{self.seed}', '*.pt')
        elif self.args.save_strategy == 'best':
            if val_loss < self.best_val_loss: 
                self.best_val_loss = val_loss         
                torch.save(
                    self.raw_model.state_dict(),
                    f'{self.args.out_dir}/best/{round(self.best_val_loss,3)}_seed{self.seed}_batch{self.train_batch_now}.pt'
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
            logits, loss = self.model(*tuple(data))
        
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

    def _run_batches(self, is_train=False):
        if is_train:
            total = self.args.eval_interval
            desc = f'[GPU{self.local_rank}]: Trianing batch {self.train_batch_now} - {self.train_batch_now + self.args.eval_interval}'
            dataloader = self.dataloader_dict['train']
            dataloader.sampler.set_batch(self.train_batch_now, True)     
        else:
            total = self.args.eval_batch_num
            desc = f'[GPU{self.local_rank}]: Calculating eval loss'
            dataloader = self.dataloader_dict['val']
            dataloader.sampler.set_batch(self.val_batch_now, False)     

        self.optimizer.zero_grad()        
        losses, visited_data = [], []
        new_lr = new_wd = grad_norm = 0
        with tqdm(total=total, desc=desc, position=self.local_rank) as pbar:
            for i, batch in enumerate(dataloader):
                batch = [x.to(self.local_rank) for x in batch]
                data, dataset_idxs = batch[:-1], batch[-1]
                visited_data.append(dataset_idxs)

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
        
        if is_train: self.train_batch_now += total
        else: self.val_batch_now += total

        batches_loss = np.array(losses).mean()
        return batches_loss, new_lr, new_wd, grad_norm, batch_token, visited_data

    def train(self):    
        early_stop_signal = torch.tensor([0,]).to(self.local_rank)
        if self.local_rank == 0:
            dataset_visited_cnt = np.zeros(len(self.dataset_dict['train']), dtype=np.int8)

        # start training
        for batch in range(self.batch_start, self.args.train_iters + 1, self.args.eval_interval):  
            self.train_batch_now = batch
            wandb_log_dict = {}

            # Save snapshot before the batch training process, avoid overlap when resuming from the snapshot
            if self.args.save_snapshot and batch % self.args.save_interval == 0 and batch != 0 and self.local_rank == 0 :
                self._save_snapshot()
            
            # Calculate validation losses at specified batch intervals
            if (batch % self.args.eval_interval == 0 or batch == self.args.train_iters) and (batch != 0 or not self.args.skip_first_eval):
                self.model.eval()
                with torch.no_grad():
                    val_loss, _, _, _, _, _ = self._run_batches(is_train=False)
                    wandb_log_dict.update({"losses/val_loss": val_loss})                    

            # exit point
            if batch == self.args.train_iters:
                break

            # training process
            self.model.train()
            start_time = time.time()
            train_loss, new_lr, new_wd, grad_norm, batch_token, visited_data = self._run_batches(is_train=True)
            self.trained_time += time.time() - start_time

            wandb_log_dict.update({
                "info/batch": batch,
                'info/lr': new_lr,
                'info/wd': new_wd,
                'info/grad_norm': grad_norm,
                'info/batch_token': batch_token,
                "losses/train_loss": train_loss, 
                'info/trained_time': self.trained_time,
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

                    # early-stopping and save-ckpt by val_loss 
                    if key == "losses/val_loss":
                        val_loss = wandb_log_dict[key]
                        self.early_stopping(val_loss)
                        if self.args.save_ckpt and ((self.args.save_strategy == 'best' and batch != 0) or (self.args.save_strategy == 'interval' and batch % self.args.save_interval == 0)):
                            self._save_checkpoint(val_loss=val_loss)

                # update dataset visited info
                visited_part = dataset_visited_cnt[dataset_visited_cnt!=0]
                wandb_log_dict.update({f'info/visited_ratio': len(visited_part)/len(dataset_visited_cnt)})   
                wandb.run.summary[f"info/visited_times"] = np.mean(visited_part)
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