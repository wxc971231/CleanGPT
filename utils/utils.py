import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import time
import random
import shutil
import numpy as np
import torch
file_path = os.path.dirname(os.path.abspath(__file__))

def create_folder_if_not_exist(floder_path):
    os.makedirs(floder_path, exist_ok=True)

def create_folder_overwrite_if_exist(floder_path):
    if os.path.exists(floder_path):
        shutil.rmtree(floder_path)    
    create_folder_if_not_exist(floder_path)

def create_file_if_not_exist(file_path):
    try:
        with open(file_path, 'a') as file:
            pass
    except FileNotFoundError:
        floder_path = file_path[:file_path.rfind('/')]
        create_folder_if_not_exist(floder_path)
        with open(file_path, 'w') as file:
            pass
        time.sleep(1)

def clean_print(str:str, rank:int=0, prefix:str=''):
    if rank == 0:
        str = prefix + '\t' + str if prefix != '' else str
        print(str)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # CUDA 的启动阻塞模式，设置为 1 时 CUDA 的所有操作将变为同步模式
    # 这确保在发生错误时，错误信息能够准确地指向问题所在的代码行，对于调试 CUDA 相关问题非常有用
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # 设置 CUBLAS 的工作空间大小，"":16:8" 表示使用 16KB 的工作空间和 8 个工作区
    # 这对于确保某些操作的确定性和性能优化可能是必要的
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    '''
    # there are some operation do not have a deterministic implementation, 
    # so actually the experiment result can't be complete reproduction
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    '''