import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(base_path)

import torch
from tqdm import tqdm
from utils.utils_model import load_model

def eval_score_adder(model, tokenizer, dataloader, total, desc=''):
    local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
    ndigit = tokenizer.ndigit

    results = []
    with tqdm(total=total, desc=desc, position=local_rank) as pbar:
        for batch in dataloader:
            x = batch[0].to(local_rank)     # (batch_size, seql_len)
            d1d2 = x[:, :ndigit*2]          # (batch_size, ndigit*2)
            d1d2d3, _ = model.generate(d1d2, ndigit+1, do_sample=False)

            correct = tokenizer.decode(d1d2, d1d2d3)    # (batch_size, )
            results.append(correct)

            correct_all = torch.vstack(results).to(torch.float32)
            correct_rate = correct_all.mean().item()
            pbar.set_postfix({'correct_rate': f'{correct_rate:.2f}'})
            pbar.update(len(correct))
            
            if correct_all.numel() >= total:
                break

    return correct_rate