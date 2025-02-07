import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import os
import torch
import wandb
import math
import time
import pprint
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP
from thop import profile
from typing import Any, Dict, List

from utils.utils import set_seed, clean_print
from data.data import AutoRegressDataset, build_dataloader_AR
from train.scheduler import EarlyStopping, OptimizerParamScheduler
from model.NanoGPT import NanoGPT, NanoGPTConfig

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    trained_time: float     # for total training time
    latest_epoch: int       # for consistent optimizer-para schedule
    best_eval_loss: float   # for early_stopping
    wandb_id: str           # for resuming wandb log

class Trainer:
    def __init__(self, args, seed, wandb_id, dataset_dict:dict):
        self.args = args
        self.seed = seed
        self.wandb_id = wandb_id
        # self.dataset_train = dataset_dict['train']
        # self.dataset_val = dataset_dict['val']
        # self.dataset_test = dataset_dict['test']
        self.dataset_dict = dataset_dict
        self.snapshot_path = f'{self.args.out_dir}/snapshot_seed{seed}.pt'
        self.world_size = int(os.environ.get("WORLD_SIZE", default='1'))
        self.local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
        set_seed(seed)   

        # AMP setting
        if args.use_amp:        
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            self.ctx = torch.amp.autocast(device_type=self.local_rank, dtype=amp_dtype)
            self.scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16)) # bfloat16 doesn't need GradScaler cause it can be calculated in hardware drictly   
        else:
            self.ctx = nullcontext()                                                    # null context
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)                      # no-op

        # init model
        self._build_model()
        
        # initialize train states
        self.early_stopping = EarlyStopping(patience=args.early_stopping_patience, delta=args.early_stopping_delta)
        self.optimizer = self.raw_model.transformer.configure_optimizers()  
        self.scheduler = OptimizerParamScheduler(self.args, self.optimizer)    
        self.best_eval_loss = None
        self.epoch_start = None
        self.epoch_now = None
        self.current_grad_accum_step = None
        self.trained_time = None

        self.grad_batch_cnt = 1         # for grad_accum
        self.model_macs_per_batch = 0

        # data stuff
        self.dataloader_dict = self._prepare_dataloader(dataset_dict)        

        # check MACs and params quantity
        model_macs_per_batch_to_share = torch.tensor([0,], dtype=torch.float32).cuda()
        if self.local_rank == 0:
            def _get_dummy_data():
                rl_task_input, batch_data_info, batch_raw_obs = next(self.dataloader_train.__iter__())
                self.dataloader_train.sampler.reset()
                rl_task_input.to(device=self.local_rank)
                for i in range(len(batch_raw_obs)):
                    for k in batch_raw_obs[i].keys():
                        batch_raw_obs[i][k] = batch_raw_obs[i][k].to(device=self.local_rank)
                batch_dataset_name = [info[0] for info in batch_data_info]
                dummy_data = {'tasks_input': rl_task_input, 'compute_loss': True, 'mems': None, 'batch_dataset_name': batch_dataset_name, 'batch_raw_obs': batch_raw_obs}
                return dummy_data
            
            def _get_dummy_model():
                if args.model == 'transformer_xl':
                    dummy_model = TransformerXL(args).to(self.local_rank)
                    dummy_model.same_length = False
                elif args.model == 'llama':
                    dummy_model = TrajLlama(args).to(self.local_rank)
                else:
                    raise NotImplementedError
                return dummy_model
            
            def _get_parameter_size_in_gb(param):
                return param.numel() * param.element_size() / 1024 / 1024 / 1024
            
            # get params quantity by count directly 
            dummy_data = _get_dummy_data()
            dummy_model = _get_dummy_model()
            total_block_params = sum(p.numel() for n, p in dummy_model.named_parameters() if 'embedding' not in n and 'encoding' not in n and n != 'lm_head')
            total_size_gb = sum(_get_parameter_size_in_gb(p) for p in dummy_model.parameters())
            #for name, param in self.named_parameters():
            #    param_size_gb = _get_parameter_size_in_gb(param)
            #    print(f"Layer: {name} \t| Parameters: {param.numel()} \t| Size: {param_size_gb:.2f} GB \t| Data Type: {param.dtype}")

            # get macs with thop lib 
            dummy_model = _get_dummy_model()
            macs, _ = profile(dummy_model, tuple(dummy_data.values()), verbose=False)
            model_macs_per_batch = macs * self.world_size
            print(f"Total block params:  \t{total_block_params}")
            print(f"Total parameter size:\t{total_size_gb:.2f} GB")
            print(f'MACs per batch data: \t{int(model_macs_per_batch /1e6)} M')
            model_macs_per_batch_to_share = torch.tensor([model_macs_per_batch,], dtype=torch.float32).cuda()    
        dist.barrier()
        dist.broadcast(model_macs_per_batch_to_share, src=0)
        self.model_macs_per_batch = model_macs_per_batch_to_share.item()
        # Mixed Precision Training
        if self.args.use_amp: 
            self.scaler = torch.cuda.amp.GradScaler()

    def _build_model(self):
        if self.args.model == 'NanoGPT':
            model_args = {key: getattr(self.args, key) for key in [
                'n_position', 'n_layer', 'n_head', 'n_embd', 'vocab_size', 'dropout', 'bias', 'weight_tie'
            ]}
            gptconf = NanoGPTConfig(**model_args)
            self.model = NanoGPT(gptconf).to(self.local_rank).train()
        else:
            raise Exception(f'{self.args.model} is not support currently')
        
        # init from latest snapshot or from scratch
        if self.args.init_from is None:
            try:
                # try to resume training from the latest snapshot
                snapshot_path = f'{self.args.out_dir}/snapshot_seed{self.seed}.pt'
                snapshot_data = torch.load(snapshot_path, map_location=self.local_rank)
                snapshot = Snapshot(**snapshot_data)
                self.model.load_state_dict(snapshot.model_state)
                self.optimizer.load_state_dict(snapshot.optimizer_state)
                self.scheduler.load_state_dict(snapshot.scheduler_state)
                self.current_grad_accum_step = self.scheduler.get_ga_step()
                self.best_eval_loss = snapshot.best_eval_loss
                self.epoch_start = snapshot.latest_epoch  
                self.epoch_now = snapshot.latest_epoch
                self.wandb_id = snapshot.wandb_id
                self.trained_time = snapshot.trained_time
                clean_print(f"Resuming training from snapshot at Epoch [{self.epoch_now}] with grad_accum_step [{self.current_grad_accum_step}]", self.local_rank)

            except FileNotFoundError:    
                # if no snapshot found, start from scratch
                self.best_eval_loss = float('inf')
                self.epoch_start = 0
                self.epoch_now = 0
                self.current_grad_accum_step = self.args.ga_begin
                self.trained_time = 0
                clean_print(f"Snapshot not found. Training model from scratch with grad_accum_step={self.current_grad_accum_step}", self.local_rank)
            
        # init from pretrained_ckpt to finetune
        elif self.args.init_from.endswith('.pt'):
            self.model.load_state_dict(torch.load(self.args.init_from, map_location=self.local_rank))
            clean_print(f"Pretrained model [{self.args.init_from}] loaded.\nFine-tuning from scratch with grad_accum_step [{self.current_grad_accum_step}]", self.local_rank)

        # special for NanoGPT, init from huggingface GPT2 ckpt
        elif self.args.model == 'NanoGPT' and self.args.init_from.startswith('gpt2'):
            override_args = dict(dropout=self.args.dropout) # only dropout can be overwritten
            self.model = self.model.from_pretrained(self.args.init_from, override_args)
            # override the created config params, so we can store them into checkpoint correctly
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                setattr(self.args, getattr(self.model.config, k))
            clean_print(f"Resuming training from OpenAI GPT-2 weights: [{self.args.init_from}]", self.local_rank)
        else:
            raise Exception(f"The model can conly be init from a .pt file path or a gpt2-model name (when model=NanoGPT), instead of {self.args.init_from}")

        # crop down the model block size if desired, using model surgery
        if self.args.model == 'NanoGPT' and self.args.n_position < self.model.config.n_position:
            self.model.crop_block_size(self.args.n_position)
            clean_print(f'The input-seq-length has been crop to [{self.args.n_position}] as args setting', self.local_rank)

        # Wrap the model DDP, which synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
    def _prepare_dataloader(self, dataset_dict:dict):
        dataloader_dict = {}
        for data_type, dataset in dataset_dict.items():
            dataloader = build_dataloader_AR(
                self.args, dataset, is_eval=(data_type != 'train'), 
                current_epoch=self.epoch_now, seed=self.seed, device=self.local_rank
            )
            dataloader_dict[data_type] = dataloader

        if self.local_rank == 0:
            print('-'*35)
            print(f'valid Dataset size:         {len(dataset_dict['val'])}')
            print(f'valid sample num per epoch: {len(dataloader['val'].sampler)}')
            print(f'train Dataset Size:         {len(dataset_dict['train'])}')
            print(f'train sample per epoch:     {self.world_size * len(len(dataloader['train']).sampler)}')
            print(f'train sample all epoch:     {self.args.train_iters * self.world_size * len(dataloader['train'].sampler)}')
            print('-'*35)

        return dataloader_dict

    
    def _save_snapshot(self):
        snapshot = Snapshot(
            model_state=self.raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            latest_epoch=self.epoch_now,
            best_retrun=self.best_retrun,
            best_eval_loss=self.best_eval_loss,
            trained_time=self.trained_time,
            wandb_id=self.wandb_id
        )
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.snapshot_path)
        #print(f"Snapshot saved at epoch {self.epoch_now}")

    def _save_checkpoint(self, ave_return=-1e5, eval_loss=1e5):
        if self.args.save_strategy == 'interval':
            torch.save(
                self.raw_model.state_dict(),
                f'{self.args.out_dir}/interval/{self.seed}/{round(ave_return,2)}_seed{self.seed}_epoch{self.epoch_now}.pt'
            )
        elif self.args.save_strategy == 'best':
            if not self.args.is_obs_pretrain and ave_return > self.best_retrun:
                self.best_retrun = ave_return
                torch.save(
                    self.raw_model.state_dict(),
                    f'{self.args.out_dir}/best/{round(self.best_retrun,3)}_seed{self.seed}_epoch{self.epoch_now}.pt'
                )
            if self.args.is_obs_pretrain and eval_loss < self.best_eval_loss: 
                self.best_eval_loss = eval_loss         
                torch.save(
                    self.raw_model.state_dict(),
                    f'{self.args.out_dir}/best/{round(self.best_eval_loss,3)}_seed{self.seed}_epoch{self.epoch_now}.pt'
                )
        else:
            raise NotImplementedError

    def _run_batch(self, rl_task_input, batch_dataset_name, batch_raw_obs, is_train=False) -> float:
        #with torch.set_grad_enabled(is_train), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(self.args.use_amp)):
        _, loss, loss_datasets, _ = self.model(rl_task_input, batch_dataset_name=batch_dataset_name, batch_raw_obs=batch_raw_obs)
        
        new_lr = None
        total_norm = None
        if is_train:
            assert self.args.use_amp is False   # NOTE(wxc): current implement of use_amp hurt the performence too much
            if self.args.use_amp: 
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self.args.clip_grad != 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_ave = loss / self.current_grad_accum_step
                if self.grad_batch_cnt % self.current_grad_accum_step != 0:
                    with self.model.no_sync():
                        loss_ave.backward()
                else:
                    loss_ave.backward()
                    if self.args.clip_grad != 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    new_lr, grad_accum_step = self.scheduler.step(self.current_grad_accum_step)
                    self.current_grad_accum_step = grad_accum_step
                    self.grad_batch_cnt = 0
            self.grad_batch_cnt += 1

        return loss.item(), loss_datasets, new_lr, total_norm

    def _run_epoch(self, is_train=False):
        epoch_losses = []
        epoch_losses_dataset = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
        data_info = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}
        desc = f'Trianing Epoch {self.epoch_now}' if is_train else 'Calculating eval loss'
        total = self.args.batch_num if is_train else self.args.eval_batch_num
        dataloader = self.dataloader_train if is_train else self.dataloader_val
        dataloader.sampler.set_epoch(self.epoch_now, is_train)  # when using DistributedSampler, its necessary to set_epoch to shuffle
        self.optimizer.zero_grad()

        desc = f'[GPU{self.local_rank}]: ' + desc
        new_lr = 0
        total_norm = 0
        with tqdm(total=total, desc=desc, position=self.local_rank) as pbar:
            for batch in dataloader:
                # 加载 batch data
                rl_task_input, batch_data_info, batch_raw_obs = batch
                if self.args.auto_batch_len and rl_task_input.seq_len.max() < self.args.n_position:
                    rl_task_input.apply(lambda x: x[:, :rl_task_input.seq_len.max()] if isinstance(x, torch.Tensor) and x.dim() == 2 else x)
                    assert (rl_task_input.tensor_seq[:,-1]==self.args.special_tokens['<|>']).sum() >= 1

                # 记录数据集访问信息
                batch_dataset_name = []
                for dataset_name, dataset_idx in batch_data_info:
                    data_info[dataset_name].append(dataset_idx)
                    batch_dataset_name.append(dataset_name)

                # run batch
                rl_task_input.to(device=self.local_rank)
                loss, loss_datasets, lr, norm = self._run_batch(rl_task_input, batch_dataset_name, batch_raw_obs, is_train)
                new_lr = lr if lr is not None else new_lr
                total_norm = norm.item() if norm is not None else total_norm

                # 记录总损失和各数据集上损失信息
                epoch_losses.append(loss)
                for dataset_name, dataset_loss in loss_datasets.items():
                    if dataset_loss != 0:
                        epoch_losses_dataset[dataset_name].append(dataset_loss.item())

                # 更新进度条
                pbar.set_postfix({
                    'total norm': '{:.4f}'.format(total_norm),
                    'batch_token': '{:.4f}'.format(self.current_grad_accum_step * self.args.batch_size * self.world_size * self.args.n_position/1e6),
                    'loss':'{:.2f}'.format(loss), 
                    'ave loss (latest 20)': '{:.2f}'.format(np.array(epoch_losses[-20:]).mean()),
                })
                pbar.update()
                
                # 归集部分 batch 运行信息上传到 wandb
                batch_info_tensor = torch.tensor([new_lr, total_norm]).to(self.local_rank)
                batch_info_gather_list = [torch.zeros_like(batch_info_tensor) for _ in range(self.world_size)]
                dist.barrier()
                dist.gather(
                    batch_info_tensor,
                    batch_info_gather_list if self.local_rank == 0 else None, 
                    dst = 0
                )                
                if self.local_rank == 0:
                    log_value_tensor = torch.mean(torch.stack(batch_info_gather_list), axis=0, dtype=torch.float32)
                    batch_log_dict = {
                        'info/total_norm': log_value_tensor[1].item(),
                        'info/batch_token': self.current_grad_accum_step * self.args.batch_size * self.world_size * self.args.n_position/1e6
                    }
                    wandb.log(batch_log_dict)

        epoch_loss = np.array(epoch_losses).mean()
        epoch_loss_dataset = {name: 0 if len(losses) == 0 else np.array(losses).mean() for name, losses in epoch_losses_dataset.items()}
        return epoch_loss, epoch_loss_dataset, new_lr, data_info

    def train(self):    
        if self.local_rank == 0:
            data_visited_cnt = {dataset.dataset_name: np.zeros(len(dataset), dtype=np.int8) for dataset in self.dataset_train.datasets}   
        early_stop_signal = torch.tensor([0,]).to(self.local_rank)

        # start training
        for epoch in range(self.epoch_start, self.args.train_iters + 1):  
            self.epoch_now = epoch
            cst_ave_return = 0
            log_dict = {}

            # Save snapshot before the epoch training process, so that there is no overlap when recover from the snapshot
            if self.args.save_snapshot and epoch % self.args.snapshot_save_interval == 0 \
                and self.local_rank == 0 and epoch != 0 :
                self._save_snapshot()
            
            # Calculate validation losses at specified epoch intervals
            if (epoch % self.args.eval_interval == 0 or epoch == self.args.train_iters) and (epoch != 0 or not self.args.skip_first_eval):
                self.model.eval()
                with torch.no_grad():
                #with torch.inference_mode():
                    # validation losses
                    self.raw_model.transformer.same_length = False               # use normal context length when loss calculating (TransformerXL back bone)
                    eval_loss, eval_loss_dataset, _, _ = self._run_epoch(is_train=False)
                    log_dict.update({"losses/eval_loss": eval_loss})
                    log_dict.update(
                        {f'eval_{dataset_name[:-3]}/eval_loss': loss
                        for dataset_name, loss in eval_loss_dataset.items()}
                    )

            # Evaluate policy performance at specified epoch intervals
            if (epoch % self.args.eval_policy_interval == 0 or epoch == self.args.train_iters) and (epoch != 0 or not self.args.skip_first_eval):
                self.model.eval()
                with torch.inference_mode():  
                    self.raw_model.transformer.same_length = self.args.use_mem   # use fixed context length when rollout with mem (TransformerXL back bone)
                    cst_returns = []
                    for setting, eval_func in self.eval_setting.items():
                        epi_return, epi_obj, epi_safe, epi_time = eval_func()
                        log_dict.update(
                            {f'eval_{dataset_name[:-3]}/return_AM({setting})': np.mean(epi_return[dataset_name]['AM'])
                            for dataset_name in self.args.eval_dataset_names}
                        )
                        log_dict.update(
                            {f'eval_{dataset_name[:-3]}/return_DB1({setting})': np.mean(epi_return[dataset_name]['DB1'])
                            for dataset_name in self.args.eval_dataset_names}
                        )
                        log_dict.update(
                            {f'eval_{dataset_name[:-3]}/obj({setting})': np.mean(epi_obj[dataset_name])
                            for dataset_name in self.args.eval_dataset_names}
                        )
                        log_dict.update(
                            {f'eval_{dataset_name[:-3]}/safe({setting})': np.mean(epi_safe[dataset_name])
                            for dataset_name in self.args.eval_dataset_names}
                        )
                        log_dict.update(
                            {f'eval_{dataset_name[:-3]}/time({setting})': np.mean(epi_time[dataset_name])
                            for dataset_name in self.args.eval_dataset_names}
                        )

                        if setting.endswith('constraint'):
                            cst_returns.extend([v['AM'] for v in epi_return.values()])  # 用 AM return 作为 ckpt 质量指标
                    cst_ave_return = np.mean(cst_returns)
                    
            if epoch == self.args.train_iters:
                break

            # one training epoch
            self.model.train()
            self.raw_model.transformer.same_length = False               # use normal context length when loss calculating (TransformerXL back bone)
            start_time = time.time()
            train_loss, train_loss_dataset, new_lr, data_info = self._run_epoch(is_train=True)
            self.trained_time += time.time() - start_time
            log_dict.update(
                {f'eval_{dataset_name[:-3]}/train_loss': loss
                for dataset_name, loss in train_loss_dataset.items()}
            )
            log_dict.update({
                "losses/train_loss": train_loss, 
                'info/trained_time': self.trained_time,
                'info/MACs': self.model_macs_per_batch * self.args.batch_num * (self.epoch_now+1),
            })
            if new_lr is not None:
                log_dict.update({'info/lr': new_lr})

            # log train data if necessary
            if self.logger is not None and self.args.traindata_logger:
                logged_data = self.dataset_train.get_logged_data()
                for env_name in self.args.eval_env_names:
                    self.logger[env_name].log_data(logged_data, seed=self.seed, is_train=True)

            #import pprint
            #pp = pprint.PrettyPrinter(indent=4)
            #print('='*10+f'{self.local_rank}'+'='*10)
            #pp.pprint({k: v for k,v in log_dict.items()})
            #print()
            
            # gather info from all GPU
            log_value_tensor = torch.tensor([log_dict[k] for k in sorted(log_dict)] + [cst_ave_return,]).to(self.local_rank)
            gather_list = [torch.zeros_like(log_value_tensor) for _ in range(self.world_size)]
            dist.barrier()
            dist.gather(
                log_value_tensor,
                gather_list if self.local_rank == 0 else None, 
                dst=0
            )

            # gather data info from all GPU
            dataset_info = [len(data_info[dataset_name]) for dataset_name in self.args.eval_dataset_names]
            for dataset_name in self.args.eval_dataset_names:
                dataset_info += data_info[dataset_name]

            dataset_info_tensor = torch.tensor(dataset_info).to(self.local_rank)
            dataset_info_gather_list = [torch.zeros_like(dataset_info_tensor) for _ in range(self.world_size)]
            dist.barrier()
            dist.gather(
                dataset_info_tensor,
                dataset_info_gather_list if self.local_rank == 0 else None, 
                dst = 0
            )

            if self.local_rank == 0:
                data_info_gather = {dataset_name: [] for dataset_name in self.args.eval_dataset_names}   
                for info in dataset_info_gather_list:
                    info_len, info = info[:len(self.dataset_train.datasets)], info[len(self.dataset_train.datasets):]
                    for l, dataset_name in zip(info_len, self.args.eval_dataset_names):
                        data_info_gather[dataset_name].append(info[:l])
                        info = info[l:]
                
                epoch_data_num = 0
                for dataset_name in self.args.eval_dataset_names:
                    dataset_idxs = torch.cat(data_info_gather[dataset_name])        # 合并各个卡在当前 epoch 中访问的数据索引
                    #unique_idxs, _ = torch.unique(dataset_idxs, return_inverse=True)
                    #assert len(unique_idxs) == dataset_idxs.numel()                # 各个卡访问的索引应该没有重叠（混合数据集时不一定）
                    epoch_data_num += len(dataset_idxs)
                    data_visited_cnt[dataset_name][dataset_idxs.cpu()] += 1
                assert epoch_data_num >= self.args.batch_size * self.args.batch_num # 多卡无法均分数据时默认向上取整

            # only do file saving and logging at rank0
            if self.local_rank == 0:
                log_value_tensor = torch.mean(torch.stack(gather_list), axis=0, dtype=torch.float32)
                cst_ave_return = log_value_tensor[-1].item()

                # save ckpt for policy learning
                if self.args.save_ckpt and not self.args.is_obs_pretrain and epoch != 0 and epoch % self.args.save_interval == 0:
                    #assert cst_ave_return != 0
                    self._save_checkpoint(ave_return=cst_ave_return)

                # log to wandb
                for i, key in enumerate(sorted(log_dict)):
                    log_dict[key] = log_value_tensor[i].item()
                    if key == 'losses/eval_loss':
                        self.early_stopping(log_value_tensor[i].item())
                        # save ckpt for pretraining
                        if self.args.save_ckpt and self.args.is_obs_pretrain and epoch != 0:
                            self._save_checkpoint(eval_loss=log_value_tensor[i].item())

                #print('='*10+f'summary'+'='*10)
                #pp.pprint({k: v for k,v in log_dict.items()})
                #print()

                log_dict.update({"info/epoch": epoch})
                for dataset_name in self.args.eval_dataset_names:
                    dataset_visited_cnt = data_visited_cnt[dataset_name]
                    visited_part = dataset_visited_cnt[dataset_visited_cnt!=0]

                    log_name = dataset_name[:dataset_name.find('_')]
                    # 训练数据集被访问比例
                    log_dict.update({f'info/{log_name}_ratio': len(visited_part)/len(dataset_visited_cnt)})
                    # 训练数据集中被访问子集的尺寸
                    wandb.run.summary[f"info/{log_name}_visited"] = len(visited_part)           
                    # 训练数据集访问计数            
                    wandb.run.summary[f"info/{log_name}_num"] = np.sum(visited_part)
                    # 训练数据集中被访问子集的访问次数
                    wandb.run.summary[f"info/{log_name}_times"] = np.mean(visited_part)
                wandb.log(log_dict)

                # early stopping
                if self.args.use_early_stopping and self.early_stopping.early_stop:
                    early_stop_signal = torch.tensor([1,]).to(self.local_rank)
                    dist.broadcast(tensor=early_stop_signal, src=0)
            
            if early_stop_signal:
                print(f'[GPU{self.local_rank}] Early Stop')
                break