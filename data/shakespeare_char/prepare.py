"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 下载 tiny shakespeare 数据集
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 获取所有唯一字符，构成 Vocab Table
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# 构造双射 ids <-> char
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]             # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l])    # decoder: take a list of integers, output a string

# 划分训练集和验证集
n = len(data)
train_data = data[:int(n*0.8)]
val_data = data[int(n*0.8):int(n*0.9)]
test_data = data[int(n*0.9):]

# 原始数据编码为 ids 序列
train_ids = encode(train_data)
val_ids = encode(val_data)
test_data = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_data):,} tokens")

# 导出为二进制文件 .bin
train_ids = np.array(train_ids, dtype=np.int64)
val_ids = np.array(val_ids, dtype=np.int64)
test_ids = np.array(test_data, dtype=np.int64)
np.save(os.path.join(os.path.dirname(__file__), 'train.npy'), train_ids)
np.save(os.path.join(os.path.dirname(__file__), 'val.npy'), val_ids)
np.save(os.path.join(os.path.dirname(__file__), 'test.npy'), test_ids)

# 保存 meta information，之后用于编码/解码
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters: 1,115,394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 892,315 tokens
# val has 111,539 tokens 
# test has 111,540 tokens
