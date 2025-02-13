import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(base_path)

import glob
import pickle
import json
import torch
from model.NanoGPT import NanoGPT, NanoGPTConfig
from utils.utils import remove_compiled_prefix

def load_model(out_path=None):
    """ load trained model from out_path """
    # load training config
    config = json.load(open(f"{out_path}/config.json", "r"))

    # create model
    if config['model'] == 'NanoGPT':
        model_args = {k: config[k] for k in [
            'n_position', 'n_layer', 'n_head', 'n_embd', 'vocab_size', 'dropout', 'add_bias', 'weight_tying'
        ]}
        gptconf = NanoGPTConfig(**model_args)
        model = NanoGPT(gptconf)
    else:
        raise Exception(f"{config['model']} is not support currently")

    # load best checkpoint
    ckpt_dir = os.path.join(f'{out_path}/best', '*.pt') if config['save_strategy'] == 'best' else \
                os.path.join(f"{out_path}/interval/{config['seed'][0]}", '*.pt')
    ckpt_files = glob.glob(ckpt_dir)
    ckpt_files = sorted(ckpt_files, key=lambda x: float(os.path.basename(x).split('_')[0]))
    best_ckpt_path = ckpt_files[0]
    ckpt_model_state = torch.load(best_ckpt_path, map_location=f"cuda:0")
    model.load_state_dict(remove_compiled_prefix(ckpt_model_state))

    # load tokenizer & decoder
    meta_path = f"{base_path}/data/{config['dataset']}/meta.pkl"
    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)
        tokenizer = meta_data['stoi']
        decoder = meta_data['itos']

    return model, tokenizer, decoder
    
def generate_text(model, tokenizer, decoder, prompt, max_length=50, piece_len=100, temperature=1.0, top_k=None, device='cuda:0'):
    ''' Generate text from prompt with streaming output '''
    res_str = prompt
    print(res_str, end='', flush=True)
    while len(res_str) < max_length:
        token_seq = torch.tensor([tokenizer[s] for s in res_str])[None,:].to(device)
        raw_res = model.generate(token_seq, piece_len, temperature, top_k).tolist()
        full_gen_str = ''.join([decoder[i] for i in raw_res[0]])
        new_part = full_gen_str[len(res_str):]
        print(new_part, end='', flush=True)
        res_str = full_gen_str

def main():
    # setting
    device = 'cuda:0'
    out_path = f"{base_path}/out/shakespeare-char_1024_384_4_6_compiled_ampd"
    prompt = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate:"
    temperature=1.0
    top_k=None
    piece_len = 50
    max_length = 1500

    # load model, tokenizer and decoder
    model, tokenizer, decoder = load_model(out_path)
    model = model.to(device).eval()

    # generate text & flow printing
    with torch.inference_mode():
        generate_text(model, tokenizer, decoder, prompt, max_length, piece_len, temperature, top_k, device)

if __name__ == "__main__":
    main()