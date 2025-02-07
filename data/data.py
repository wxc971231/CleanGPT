import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..'))
sys.path.append(base_path)

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class AutoRegressDataset(Dataset):
    def __init__(self, args:argparse.Namespace, data_path:str): 
        self.n_position = args.n_position
        self.data_path = data_path
            
    def __len__(self):
        # 自回归模型的标签是样本后错一位，长为 n 的连续序列中共能构造出长 m 的序列样例 n-m 个
        data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        return len(data) - self.n_position

    def __getitem__(self, idx):
        # 此处 Dataset 对象只返回索引，在 collate_fn 中构成 batch 数据，避免将整个数据集加载到内存中
        return idx

def collate_fn(batch, data_path, n_position, device=0):
    # np.memmap 是一种内存映射文件的方式，可在不将整个文件加载到内存中的情况下访问大文件
    # 为每个 batch 构造新的 np.memmap 对象，以免出现内存泄漏问题 (https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122)
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    idxs = torch.tensor(batch)
    x = torch.stack([torch.from_numpy((data[i:i+n_position]).astype(np.int64)) for i in idxs])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+n_position]).astype(np.int64)) for i in idxs])
    
    if device == 'cpu':
        x, y = x.to(device), y.to(device)
    else:
        # pin_memory() 将张量固定在内存中，而后可通过 DMA 通道直接传输到 GPU，避免 CPU-GPU 之间的内存拷贝
        # non_blocking=True 允许异步传输数据，主线程不会阻塞，从而可以并行执行其他操作
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    
    return x, y
        
def build_dataloader_AR(args, dataset:AutoRegressDataset, is_eval:bool=False, current_epoch:int=0, seed:int=42, device=0):
    """ Buld DDP dataloader given an input dataset. """
    if dataset is None:
        return None
    
    # The DistributedSampler automatically blocks the data and sends it to each Gpu, which can avoid data overlap
    sampler = DistributedSampler(dataset=dataset, seed=seed)

    # Set the initial epoch to ensure consistent data division, especially when resuming from snapshot
    sampler.set_epoch(current_epoch)        
    
    batch_size = args.eval_batch_size if is_eval else args.batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,                    # pin data in memory, which enable DMA transfer tube between CPU and GPU
        shuffle=False,                      # Must be False, as the DistributedSampler will handle the shuffling
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, dataset.data_path, dataset.n_position, device)
    )