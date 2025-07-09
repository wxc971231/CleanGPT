import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(base_path)

import torch
from utils.utils_model import load_model

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
    else:
        raise Exception(f"{dataset_name} is not support currently")
        
def main():
    # setting
    # device = 'cuda:0'
    # out_path = f"{base_path}/out/ShakespeareChar_1024_256_8_4"
    # prompt = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate:"
    # temperature = 0.02
    # top_k = None
    # do_sample = True
    # piece_len = 20
    # max_length = 1000

    device = 'cuda:0'
    out_path = f"{base_path}/out/TinyStory_llama_1024_512_8_10"
    prompt = "Once upon a time, "
    temperature = 0.1
    top_k = None
    do_sample = False
    piece_len = 20
    max_length = 1000

    # load model, tokenizer and decoder
    _, model, dataset_name, tokenizer, decoder = load_model(out_path)
    model = model.to(device).eval()

    # generate text & flow printing
    print('\n'+'-'* 50 + 'Generating Story' + '-'*50)
    with torch.inference_mode():
        generate_text(model, dataset_name, tokenizer, decoder, prompt, max_length, piece_len, temperature, top_k, device, do_sample)

if __name__ == "__main__":
    main()