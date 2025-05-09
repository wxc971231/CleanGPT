import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..'))
sys.path.append(base_path)

import argparse
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Union, Optional
from data.adder.prepare import AdditionDataset, AdditionTokenizer

class AutoRegressDataset(Dataset):
    def __init__(self, args:argparse.Namespace, data_path:str): 
        self.n_position = args.n_position
        self.data_path = data_path

        # use mmap_mode to avoid loading the entire dataset into memory
        data = np.load(self.data_path, mmap_mode='r', allow_pickle=True)
        self.length = len(data) - self.n_position

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):
        # Only return data-idx here, the batch data is constructed in collate_fn
        # by np.memmap to avoid loading the entire dataset into memory
        return idx % self.length

def collate_fn(batch, data_path, n_position):
    data = np.load(data_path, mmap_mode='r')
    idxs = torch.tensor(batch)
    if data.ndim == 1:
        x = torch.stack([torch.from_numpy((data[i:i+n_position]).astype(np.int64)) for i in idxs])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+n_position]).astype(np.int64)) for i in idxs])
    elif data.ndim == 2:
        x = torch.stack([torch.from_numpy((data[i,:-1]).astype(np.int64)) for i in idxs])
        y = torch.stack([torch.from_numpy((data[i,1:]).astype(np.int64)) for i in idxs])
    else:
        raise ValueError(f"data shape {data.shape} not supported")

    return x, y, idxs            

def build_dataloader(args, dataset:AutoRegressDataset, is_eval:bool=False, current_batch:int=0, seed:int=42):
    """ Buld DDP dataloader given an input dataset. """
    if dataset is None:
        return None
    
    batch_size_per_gpu = args.eval_batch_size_per_gpu if is_eval else args.batch_size_per_gpu

    # The DistributedSampler automatically blocks the data and sends it to each Gpu, which can avoid data overlap
    sampler = MyDistributedSampler(
        dataset=dataset,
        num_replicas=int(os.environ.get("WORLD_SIZE", default='1')),
        rank=int(os.environ.get("LOCAL_RANK", default='0')),
        seed=seed,
        init_batch=current_batch, 
        batch_size_per_gpu=batch_size_per_gpu,
        batch_num_per_iter=args.eval_batch_num if is_eval else args.eval_interval,
        drop_last=False
    )

    if args.dataset in ['tinystory', 'shakespeare_char']:
        collate_func = lambda batch: collate_fn(batch, dataset.data_path, dataset.n_position)
    elif args.dataset == 'adder':
        collate_func = None
    else:
        raise ValueError(f"dataset {args.dataset} not supported")

    # build DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        pin_memory=True,                    # pin data in memory, which enable DMA transfer tube between CPU and GPU
        shuffle=False,                      # Must be False, as the DistributedSampler will handle the shuffling
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_func
    )

class MyDistributedSampler(DistributedSampler):
    def __init__(
        self, dataset: Dataset, num_replicas: Optional[int] = None, 
        rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False,
        init_batch: int = 0, batch_size_per_gpu: int = 64, batch_num_per_iter: int = 64
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size_per_gpu * num_replicas

        # when self.__iter__ called, a iterator with size of macro_batch_size returned
        self.macro_batch_size = batch_num_per_iter * self.batch_size     
        self.macro_batch_num_per_epoch = len(dataset) // (self.macro_batch_size) if self.drop_last else math.ceil(len(dataset) / (self.macro_batch_size))
        self.epoch_total_samples = self.macro_batch_num_per_epoch * self.macro_batch_size
        self.batch_num_per_epoch = int(self.epoch_total_samples / self.batch_size)

        # get current epoch and data offset, which is useful for resuming training
        self.current_epoch = init_batch // self.batch_num_per_epoch
        self.current_data_offset = self.batch_size * int(init_batch % self.batch_num_per_epoch)
        self.init_data_offset = self.current_data_offset

        # generate the sample permutation order of current epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(seed + self.current_epoch)
            self.sample_idxs = torch.randperm(self.epoch_total_samples, generator=g).tolist()
        else:
            self.sample_idxs = list(range(self.epoch_total_samples))
            
    def __iter__(self):
        # clip a segment from the permutation with size of macro_batch_size
        indices = self.sample_idxs[self.current_data_offset : self.current_data_offset + self.macro_batch_size]
        indices = (np.array(indices) % len(self.dataset)).tolist()  # prevent index out of range
        assert len(indices) == self.macro_batch_size

        # re-generate the sample permutation when the current epoch is finished 
        self.current_data_offset += self.macro_batch_size
        if self.current_data_offset >= self.epoch_total_samples:
            self.current_data_offset = 0
            self.current_epoch += 1
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.current_epoch)
                self.sample_idxs = torch.randperm(self.epoch_total_samples, generator=g).tolist()
        
        # subsample
        indices = indices[self.rank : self.macro_batch_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.macro_batch_num_per_epoch

    def reset(self):
        self.current_data_offset = self.init_data_offset

    def set_batch(self, batch:int, macro_batch_now:int, is_train:bool) -> None:
        if is_train:
            # 训练集，确保每个 epoch 取数据无重叠
            assert self.current_data_offset == self.batch_size * int(batch % self.batch_num_per_epoch)
        else:
            # 验证集，每 n 个训练 epoch 执行一次，每次重新随机取数据
            g = torch.Generator()
            g.manual_seed(self.seed + macro_batch_now)
            self.sample_idxs = torch.randperm(self.epoch_total_samples, generator=g).tolist()
            self.current_data_offset = 0
