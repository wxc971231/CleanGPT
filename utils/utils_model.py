import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import glob
import pickle
import json
import torch
import argparse
from utils.utils import clean_print
from model.NanoGPT import NanoGPT, NanoGPTConfig
from model.llama import MyLlama, MyLlamaConfig
from data.adder.prepare import AdditionTokenizer
from data.multiplier.prepare import MultiplicationTokenizer

def remove_compiled_prefix(state_dict):
    # when using torch.compile, the model's name will be compiled to _orig_mod.xxx, which cause the state_dict can't be loaded directly.
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    return state_dict

def load_model(out_path=None):
    """ load trained model from out_path """
    # load training config
    config = json.load(open(f"{out_path}/config.json", "r"))
    local_rank = int(os.environ.get("LOCAL_RANK", default='0'))

    # load tokenizer & decoder
    dataset_name = config['dataset']
    meta_path = f"{base_path}/data/{dataset_name}/meta.pkl"
    if dataset_name == 'tinystory':
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
            tokenizer = meta_data['tokenizer']
            decoder = None
    elif dataset_name == 'shakespeare_char':
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
            tokenizer = meta_data['stoi']
            decoder = meta_data['itos']
    elif dataset_name == 'adder':
        tokenizer = AdditionTokenizer(config['adder_ndigit'], config['adder_format_vocab'])
        decoder = None
    elif dataset_name == 'multiplier':
        tokenizer = MultiplicationTokenizer(config['multiplier_ndigit'], config['multiplier_format_vocab'])
        decoder = None
    else:
        raise Exception(f"{dataset_name} is not support currently")

    # create model
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None else None
    if config['model'] == 'NanoGPT':
        model_args = {k: config[k] for k in ['n_position', 'n_layer', 'n_head', 'n_embed', 'vocab_size', 'dropout', 'dropout_attn', 'add_bias', 'weight_tying']}
        model_args.update({'mask_out_token': pad_token_id})
        gpt_conf = NanoGPTConfig(**model_args)
        model = NanoGPT(gpt_conf)
    elif config['model'] == 'llama':
        model_args = {k: config[k] for k in [
            'n_position', 'n_layer', 'n_q_head', 'n_kv_head', 'n_embed', 'vocab_size', 
            'rms_norm_eps', 'weight_tying', 'dropout_attn', 'add_bias', 'lr_begin',
            'adam_beta1', 'adam_beta2', 'adam_eps', 
        ]}
        model_args.update({
            'pad_token_id': pad_token_id,
            'mask_out_token': pad_token_id,
            'weight_decay': config['wd_begin'],
        })
        assert config['wd_decr_style'] == "constant"
        llama_conf = MyLlamaConfig(**model_args)
        model = MyLlama(llama_conf)
    else:
        raise Exception(f'{config["model"]} is not support currently')

    # load best checkpoint
    ckpt_dir = os.path.join(f'{out_path}/best', '*.pt') if config['save_strategy'] == 'best' else \
                os.path.join(f"{out_path}/interval/{config['seed'][0]}", '*.pt')
    ckpt_files = glob.glob(ckpt_dir)
    ckpt_files = sorted(ckpt_files, key=lambda x: float(os.path.basename(x).split('_')[0]))
    best_ckpt_path = ckpt_files[0]
    ckpt_model_state = torch.load(best_ckpt_path, map_location=f"cuda:0")
    model.load_state_dict(remove_compiled_prefix(ckpt_model_state))
    clean_print(f"Load {best_ckpt_path}", local_rank, '[Model]')

    args = argparse.Namespace(**config)
    return args, model, dataset_name, tokenizer, decoder