"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
    https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import os
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.utils import clean_print

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.add_bias)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.add_bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            clean_print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0", int(os.environ.get("LOCAL_RANK", default='0')), '[NanoGPT]')
            # causal mask to ensure that attention is only applied to the left in the input sequence
            bias = torch.tril(torch.ones(config.n_position, config.n_position)).view(1, 1, config.n_position, config.n_position)
            # invoke 'reigster_buffer' save tenosr 'bias' in model's state_dict, so that it can be moved to target device with the model
            # and the 'requires_grad' of reigsted tensor is False, so it will not be updated by optimizer
            self.register_buffer("bias", bias)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v                                     # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.add_bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.add_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.add_bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.add_bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class NanoGPTConfig:
    n_position: int = 1024
    vocab_size: int = 50304     # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    weight_tying: bool = True   # share token-embedding-vector between input token-embedding layer and output lm-head layer
    add_bias: bool = True       # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
class NanoGPT(nn.Module):
    local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
    
    def __init__(self, config, mask_out_token=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.n_position is not None
        self.config = config
        self.mask_out_token = mask_out_token

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.n_position, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.add_bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.n_position
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()                                           # (batch_size, seq_len)
        assert t <= self.config.n_position, f"Cannot forward sequence of length {t}, block size is only {self.config.n_position}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)   # (seq_len, )

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)             # (batch_size, seq_len, n_embd)
        pos_emb = self.transformer.wpe(pos)             # (seq_len, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)    # (batch_size, seq_len, n_embd)
        for block in self.transformer.h:
            x = block(x)                                
        x = self.transformer.ln_f(x)                    # (batch_size, seq_len, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)                    # (batch_size, seq_len)
            
            if self.mask_out_token is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                # mask out loss contributions from padding tokens
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none')
                mask = (targets != self.mask_out_token).float()
                loss = (loss * mask.view(-1)).sum() / mask.sum()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, n_position):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert n_position <= self.config.n_position
        self.config.n_position = n_position
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:n_position])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:n_position,:n_position]

    @classmethod
    def from_pretrained(cls, model_type, override_config=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        # only dropout can be overridden
        override_config = override_config or {} 
        assert all(k == 'dropout' for k in override_config)
        from transformers import GPT2LMHeadModel
        clean_print(f"loading weights from pretrained gpt: {model_type}", NanoGPT.local_rank, '[NanoGPT]')

        # n_layer, n_head and n_embd are determined from model_type
        config_config = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        clean_print("forcing vocab_size=50257, n_position=1024, bias=True", NanoGPT.local_rank, '[NanoGPT]')
        config_config['vocab_size'] = 50257     # always 50257 for GPT model checkpoints
        config_config['n_position'] = 1024      # always 1024 for GPT model checkpoints
        config_config['add_bias'] = True        # always True for GPT model checkpoints
        
        # we can override the dropout rate, if desired
        if 'dropout' in override_config:
            clean_print(f"overriding dropout rate to [{override_config['dropout']}]", NanoGPT.local_rank, '[NanoGPT]')
            config_config['dropout'] = override_config['dropout']

        # create a from-scratch initialized minGPT model
        config = NanoGPTConfig(**config_config)
        model = NanoGPT(config)

        # discard this mask / buffer, not a param
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]        # same, just the mask (buffer)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, args, device_type=0):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': args.wd_begin},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        clean_print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters", NanoGPT.local_rank, '[NanoGPT]')
        clean_print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters", NanoGPT.local_rank, '[NanoGPT]')
        
        # Create AdamW optimizer and use the fused version if it is available
        # the fused version of the AdamW kernel combines multiple CUDA operations into a single kernel for execution to speed up training
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type != 'cpu'
        extra_config = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=args.lr_begin, 
            betas=(args.adam_beta1, args.adam_beta2), 
            eps=args.adam_eps,
            **extra_config
        )
        clean_print(f"Using fused AdamW: {use_fused}", NanoGPT.local_rank, '[NanoGPT]')

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_token_idx=None, temperature=1.0, top_k=None, do_sample=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generated_eos = False
        for _ in range(max_new_tokens):
            # crop sequence context at n_position 
            idx_cond = idx if idx.size(1) <= self.config.n_position else idx[:, -self.config.n_position:]
            logits, _ = self(idx_cond)                              # (batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :] / temperature                 # (batch_size, vocab_size)
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # either sample from the distribution or take the most likely element
            probs = F.softmax(logits, dim=-1)                       # (batch_size, vocab_size)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)        # (batch_size, 1)

            # exit if we hit the end of the sequence token
            if eos_token_idx is not None and idx_next.item() == eos_token_idx:
                generated_eos = True
                break

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx, generated_eos
