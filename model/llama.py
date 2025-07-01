import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, LlamaModel
from transformers.models.llama import LlamaForCausalLM
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from dataclasses import dataclass, asdict

@dataclass
class MyLlamaConfig:
    n_position: int = 1024
    n_layer: int = 12
    n_q_head: int = 12
    n_kv_head: int = 12
    n_embed: int = 768
    rms_norm_eps: float = 1e-6
    dropout_attn: float = 0.0
    pad_token_id: int = -1
    weight_decay: float = 1e-3
    lr_begin: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    vocab_size: int = 50304     # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    weight_tying: bool = True   # share token-embedding-vector between input token-embedding layer and output lm-head layer
    add_bias: bool = True       # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mask_out_token: int = None  # if not None, this token will be masked out in the loss calculation, e.g. for padding tokens

class MyLlama(torch.nn.Module):
    def __init__(self, config:MyLlamaConfig):
        super().__init__()
        self.config = config
        self.n_embed = config.n_embed                               
        self.n_position = config.n_position                         
        self.n_layer = config.n_layer                               
        self.n_q_head = config.n_q_head                            
        self.n_kv_head = config.n_kv_head                           
        self.d_head = self.n_embed // self.n_q_head                 
        self.d_inner = (int(self.n_embed * 8/3 / 64) + 1) * 64      # the inner dimension of the SwiGLU, use 8/3 instead of 2 is because SwiGLU has three parameter matrices
        self.vocab_size = self.config.vocab_size

        # build llama backbone
        self.llama_config = AutoConfig.for_model(
            model_type="llama",
            vocab_size = self.vocab_size,
            hidden_size = self.n_embed,
            intermediate_size = self.d_inner,
            num_hidden_layers = self.n_layer,
            num_attention_heads = self.n_q_head,
            num_key_value_heads = self.n_kv_head,
            max_position_embeddings = self.n_position,
            rms_norm_eps = self.config.rms_norm_eps,
            pad_token_id=self.config.pad_token_id,
            bos_token_id=None,
            eos_token_id=None,
            tie_word_embeddings=self.config.weight_tying,
            attention_bias=False,
            attention_dropout=self.config.dropout_attn,
        )
        self.model = LlamaModel._from_config(self.llama_config, torch_dtype=torch.float32)
        self.lm_head = torch.nn.Linear(self.n_embed, self.vocab_size, bias=False)
        if self.config.weight_tying:
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head.weight.data.normal_(mean=0.0, std=0.02)

    def configure_optimizers(self):
        """
        utilize weight decay method to avoid overfitting
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay_parameters = get_parameter_names(self, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        decay = set([n for n, p in self.named_parameters() if (n in decay_parameters and p.requires_grad)])
        no_decay = set([n for n, p in self.named_parameters() if (n not in decay_parameters and p.requires_grad)])

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=self.config.lr_begin, 
            betas=(self.config.adam_beta1, self.config.adam_beta2), 
            eps=self.config.adam_eps
        )
        return optimizer
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        targets: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False if targets is not None else use_cache,  # no need to cache if we are training
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
        hidden_states = outputs.last_hidden_state           # (batch_size, seq_len, n_embed)

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            logits = self.lm_head(hidden_states)            # (batch_size, klen, vocab_size)

            if self.config.mask_out_token is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                # mask out loss contributions from padding tokens
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none')
                mask = (targets != self.config.mask_out_token).float()
                loss = (loss * mask.view(-1)).sum() / mask.sum()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(hidden_states[:, [-1], :])# (batch_size, 1, 1)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
    ):
        if input_ids.size(1) > self.config.n_position:
            input_ids = input_ids[:, -self.config.n_position:]

        generated = input_ids
        past_key_values = None
        generated_eos = False
        for _ in range(max_new_tokens):
            if past_key_values is None:
                input_cur = generated
            else:
                input_cur = generated[:, -1].unsqueeze(-1)

            outputs = self.model(
                input_ids=input_cur,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state       # (batch, seq, hidden)
            logits = self.lm_head(hidden_states[:, -1, :])  # (batch_size, vocab_size)
            logits = logits / temperature                   # (batch_size, vocab_size)
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)               # (batch, vocab_size)

            # generate the next token
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)        # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # early stop if we hit the eos token
            if eos_token_id is not None:
                assert len(input_ids) == 1, "eos_token_idx is only supported for batch size 1"
                if next_token.item() == eos_token_id:
                    generated_eos = True
                    break

        return generated, generated_eos
        