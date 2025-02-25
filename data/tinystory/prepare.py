import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

import pickle
import datasets
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm

def tokenize_func(text_list:List[str], max_length:str=1024, data_type:str='') -> Dict[str, List]:
    # batch encode without special bos_token <s>
    encoded_texts = tokenizer(text_list, add_special_tokens=False)
    input_ids_list = encoded_texts['input_ids']

    raw_data_list, data_length = [], []
    for input_ids in tqdm(input_ids_list, desc=f'Tokenizing {data_type} Dataset'):
        data_length.append(len(input_ids))
        ids = input_ids[:max_length]
        ids = ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (max_length - len(ids))
        assert len(ids) == max_length + 1
        raw_data_list.append(ids)
    
    # plot the data length distribution
    sorted_data_length = sorted(data_length)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data_length)), sorted_data_length)
    plt.xlabel('Index')
    plt.ylabel('Length')
    plt.title('Data Length Distribution')
    plt.savefig(f'{os.path.dirname(os.path.abspath(__file__))}/{data_type}.png')

    return raw_data_list

if __name__ == '__main__':
    # Download raw data form mirror server
    dataset_saved_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name_or_path = "noanabeshima/TinyStoriesV2"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.system(f"huggingface-cli download --repo-type dataset noanabeshima/TinyStoriesV2 --cache-dir {dataset_saved_dir}")

    # Load the dataset, only use 10% of the training data
    config = datasets.DownloadConfig(force_download=False, max_retries=100) 
    dataset_path = f'{base_path}/data/tinystory/datasets--noanabeshima--TinyStoriesV2/snapshots/50ffd6925642413b001a000bc30e1d90aed9f5da'
    train_dataset = load_dataset('json', data_files=f'{dataset_path}/TinyStoriesV2-GPT4-train.jsonl', split='train[:10%]')
    val_dataset = load_dataset('json', data_files=f'{dataset_path}/TinyStoriesV2-GPT4-valid.jsonl', split='train')

    # Use llama2 tokenizer with 32k vocab size, the llama2 version with 128k is too large
    dataset_saved_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf', cache_dir=dataset_saved_dir, force_download=False, resume_download=None)
    
    # tokenize the dataset
    seq_len = 1024
    train_ids_list = tokenize_func(train_dataset['text'], seq_len, data_type='Train')
    val_ids_list = tokenize_func(val_dataset['text'], seq_len, data_type='Val')
    train_ids = np.array(train_ids_list, dtype=np.uint16)
    val_ids = np.array(val_ids_list, dtype=np.uint16)
    print(f"train_data.shape={train_ids.shape}")
    print(f"val_data.shape={val_ids.shape}")

    # save the tokenized dataset
    np.save(os.path.join(os.path.dirname(__file__), 'train.npy'), train_ids)
    np.save(os.path.join(os.path.dirname(__file__), 'val.npy'), val_ids)

    # save meta information for decoding
    meta = {'tokenizer': tokenizer}
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # train_data.shape=(271769, 1024)
    # val_data.shape=(27629, 1024)