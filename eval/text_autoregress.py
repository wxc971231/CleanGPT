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
        model_args = {k: config[k] for k in ['n_position', 'n_layer', 'n_head', 'n_embd', 'vocab_size', 'dropout', 'add_bias', 'weight_tying']}
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
        pass
    else:
        raise Exception(f"{dataset_name} is not support currently")
    
    return model, dataset_name, tokenizer, decoder
    
def generate_text(model, dataset_name, tokenizer, decoder, prompt, max_length=50, piece_len=100, temperature=1.0, top_k=None, device='cuda:0', do_sample=False):
    ''' Generate text from prompt with streaming output '''
    res_str = prompt
    print(res_str, end='', flush=True)
    if dataset_name == 'tinystory':
        eos_token_idx = tokenizer.vocab[tokenizer.eos_token]
        while len(res_str) < max_length:
            token_seq = tokenizer.tokenize(res_str)
            token_seq = torch.tensor(tokenizer.convert_tokens_to_ids(token_seq))[None,:].to(device)
            raw_res, generated_eos = model.generate(token_seq, piece_len, eos_token_idx, temperature, top_k, do_sample)
            raw_res = raw_res[0].tolist()
            full_gen_tokens = tokenizer.convert_ids_to_tokens(raw_res)
            full_gen_str = tokenizer.convert_tokens_to_string(full_gen_tokens)
            new_part = full_gen_str[len(res_str):]
            print(new_part, end='', flush=True)
            res_str = full_gen_str

            if generated_eos:
                break

    elif dataset_name == 'shakespeare_char':
        eos_token_idx = None
        while len(res_str) < max_length:
            token_seq = torch.tensor([tokenizer[s] for s in res_str])[None,:].to(device)
            raw_res, _ = model.generate(token_seq, piece_len, eos_token_idx, temperature, top_k, do_sample)
            raw_res = raw_res.tolist()
            full_gen_str = ''.join([decoder[i] for i in raw_res[0]])
            new_part = full_gen_str[len(res_str):]
            print(new_part, end='', flush=True)
            res_str = full_gen_str
    elif dataset_name == 'adder':
        pass
    else:
        raise Exception(f"{dataset_name} is not support currently")
        
def main():
    # setting
    device = 'cuda:0'
    out_path = f"{base_path}/out/ShakespeareChar_1024_256_8_4"
    prompt = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate:"
    # out_path = f"{base_path}/out/TinyStory_1024_256_8_4"
    # prompt = "Once upon a time, "
    temperature = 0.1
    top_k = None
    do_sample = False
    piece_len = 20
    max_length = 5000

    # load model, tokenizer and decoder
    model, dataset_name, tokenizer, decoder = load_model(out_path)
    model = model.to(device).eval()

    # generate text & flow printing
    with torch.inference_mode():
        generate_text(model, dataset_name, tokenizer, decoder, prompt, max_length, piece_len, temperature, top_k, device, do_sample)

if __name__ == "__main__":
    main()