import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(base_path)

from configs.base import BaseExperimentConfig

class AdderNanoGPTConfig(BaseExperimentConfig):    
    def __init__(self):
        super().__init__()
        self.description = "NanoGPT model trained on Adder dataset"
    
    def get_args_ready(self):
        args = self.get_base_args()
        WORLD_SIZE = args.world_size
        
        # model setting
        args.model = 'NanoGPT'
        args.n_position = 1024
        args.n_layer = 4
        args.n_head = 8
        args.n_embed = 256
        args.n_inner = 4 * args.n_embed
        args.dropout = 0.0
        args.dropout_attn = 0.0
        args.init_from = None
        
        # data setting
        args.dataset = "adder"
        args.math_vocab = {'=': 10, '+': 11, 'x': 12}
        args.adder_format_vocab = {'=': args.math_vocab['='], '+': args.math_vocab['+']}
        args.adder_ndigit = 3
        args.adder_use_format = True
        
        # optimizer setting
        args.lr_begin = 0                                       
        args.lr_max = 1e-3                          # with baby networks can afford to go a bit higher
        args.lr_decay_factor = 10.0                 # min learning rate equals to (learning_rate / 10) usually
        args.lr_warmup_ratio = 0.05
        args.lr_decay_ratio = 0.95
        args.lr_decay_style = "cosine"
        args.wd_begin = 1e-3                        # with baby networks can afford to go a bit higher (1e-4 ~ 1e-2)
        args.wd_end = args.wd_begin                 # For most of situation, keep the weight decay coefficient 'constant' is suitable
        args.wd_decr_style = "constant"            
        args.ga_begin = 4                           # batch_grad_accum is used to simulate larger batch sizes              
        args.ga_end = args.ga_begin                 # with baby networks we can simply use 'constant' grad_accum_step, but for large networks sometimes increase to 2x~10x
        args.ga_incr_style = "constant"
        args.adam_beta2 = 0.99                      # make a bit bigger because number of tokens per iter is small

        # training setting
        args.batch_size_per_gpu = 128                                            # training batch_size (per GPU)
        args.batch_size = args.batch_size_per_gpu * WORLD_SIZE * args.ga_begin  # equivalent training batch_size
        args.batch_num = 64 * args.ga_begin
        args.train_iters = 256 * args.batch_num                                 # total batch_num
        args.early_stopping_patience = 6
        args.early_stopping_delta = 0
        args.clip_grad = 1.0                        # clip gradients at this value, or disable if == 0.0

        # eval setting
        args.eval_batch_num = 20
        args.eval_batch_size_per_gpu = 64
        args.eval_batch_size = args.eval_batch_size_per_gpu * WORLD_SIZE
        args.eval_interval = args.batch_num * 1     # keep frequent because we'll overfit
        args.problem_batch_num = 2
        args.problem_batch_size_per_gpu = 64
        args.total_problem_num = args.problem_batch_size_per_gpu * WORLD_SIZE * args.problem_batch_num
        args.eval_score_interval = args.batch_num * 2
        args.resample_times = 8

        # ctrl setting, which are usually changed
        args.weight_tying = True                    # tie the word embedding and softmax weights, like in GPT-2
        args.add_bias = False                       # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        args.override_opt_param_scheduler = True    # Set 'True' to override all scheduler setting, otherwise the scheduler will be set by checkpoint
        args.skip_first_eval = False                # skip the first evaluation to at batch 0
        args.wandb = True                           # use wandb to log training info
        args.use_early_stopping = True              # use the early stopping mechanism to aviod overfitting
        args.use_kvcache = False                    # use kv cache to speed up evaluation          
        args.use_amp = False                        # use automatic mixed precision (AMP) to speed up training, which may hurt the performance
        args.compile = False                        # compile the model to speed up training, only support torch 2.0+
        args.save_snapshot = True                   # save the latest traing snapshot, from which we can resume training 
        args.save_ckpt = True                       # update ckpt by save_interval and save_strategy
        args.save_ckpt_num = 3                      # the number of ckpt to save, 0 for saving all ckpt
        args.save_interval = args.batch_num * 1     # save snapshot by interval
        args.save_strategy = 'best'                 # 'best' or 'interval'
        
        # IO setting
        args.wandb_project = 'CleanGPT'
        args.exp_name = f'Adder({args.adder_ndigit}_format)' if args.adder_use_format else f'Adder({args.adder_ndigit})'
        args.exp_profile = f'{args.exp_name}_{args.model}_{args.n_position}_{args.n_embed}_{args.n_head}_{args.n_layer}'
        args.exp_profile = f'{args.exp_profile}_compiled' if args.compile else args.exp_profile
        args.exp_profile = f'{args.exp_profile}_ampd' if args.use_amp else args.exp_profile
        
        return args

class AdderLlamaConfig(BaseExperimentConfig):    
    def __init__(self):
        super().__init__()
        self.description = "llama model trained on Adder dataset"
    
    def get_args_ready(self):
        args = self.get_base_args()
        WORLD_SIZE = args.world_size
        
        # model setting
        args.model = 'llama'
        args.n_position = 1024
        args.n_layer = 4
        args.n_q_head = 8
        args.n_kv_head = 8
        args.n_embed = 256
        args.dropout = 0.0
        args.dropout_attn = 0.0
        args.init_from = None
        
        # data setting
        args.dataset = "adder"
        args.math_vocab = {'=': 10, '+': 11, 'x': 12}
        args.adder_format_vocab = {'=': args.math_vocab['='], '+': args.math_vocab['+']}
        args.adder_ndigit = 3
        args.adder_use_format = True
        
        # optimizer setting
        args.lr_begin = 0                                       
        args.lr_max = 1e-3                          # with baby networks can afford to go a bit higher
        args.lr_decay_factor = 10.0                 # min learning rate equals to (learning_rate / 10) usually
        args.lr_warmup_ratio = 0.05
        args.lr_decay_ratio = 0.95
        args.lr_decay_style = "cosine"
        args.wd_begin = 1e-3                        # with baby networks can afford to go a bit higher (1e-4 ~ 1e-2)
        args.wd_end = args.wd_begin                 # For most of situation, keep the weight decay coefficient 'constant' is suitable
        args.wd_decr_style = "constant"            
        args.ga_begin = 4                           # batch_grad_accum is used to simulate larger batch sizes              
        args.ga_end = args.ga_begin                 # with baby networks we can simply use 'constant' grad_accum_step, but for large networks sometimes increase to 2x~10x
        args.ga_incr_style = "constant"
        args.adam_beta2 = 0.99                      # make a bit bigger because number of tokens per iter is small

        # training setting
        args.batch_size_per_gpu = 128                                            # training batch_size (per GPU)
        args.batch_size = args.batch_size_per_gpu * WORLD_SIZE * args.ga_begin  # equivalent training batch_size
        args.batch_num = 64 * args.ga_begin
        args.train_iters = 256 * args.batch_num                                 # total batch_num
        args.early_stopping_patience = 6
        args.early_stopping_delta = 0
        args.clip_grad = 1.0                        # clip gradients at this value, or disable if == 0.0

        # eval setting
        args.eval_batch_num = 20
        args.eval_batch_size_per_gpu = 64
        args.eval_batch_size = args.eval_batch_size_per_gpu * WORLD_SIZE
        args.eval_interval = args.batch_num * 1     # keep frequent because we'll overfit
        args.problem_batch_num = 2
        args.problem_batch_size_per_gpu = 64
        args.total_problem_num = args.problem_batch_size_per_gpu * WORLD_SIZE * args.problem_batch_num
        args.eval_score_interval = args.batch_num * 2
        args.resample_times = 8

        # ctrl setting, which are usually changed
        args.weight_tying = True                    # tie the word embedding and softmax weights, like in GPT-2
        args.add_bias = False                       # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        args.override_opt_param_scheduler = True    # Set 'True' to override all scheduler setting, otherwise the scheduler will be set by checkpoint
        args.skip_first_eval = False                # skip the first evaluation to at batch 0
        args.wandb = True                           # use wandb to log training info
        args.use_early_stopping = True              # use the early stopping mechanism to aviod overfitting
        args.use_kvcache = False                    # use kv cache to speed up evaluation          
        args.use_amp = False                        # use automatic mixed precision (AMP) to speed up training, which may hurt the performance
        args.compile = False                        # compile the model to speed up training, only support torch 2.0+
        args.save_snapshot = True                   # save the latest traing snapshot, from which we can resume training 
        args.save_ckpt = True                       # update ckpt by save_interval and save_strategy
        args.save_ckpt_num = 3                      # the number of ckpt to save, 0 for saving all ckpt
        args.save_interval = args.batch_num * 1     # save snapshot by interval
        args.save_strategy = 'best'                 # 'best' or 'interval'
        
        # IO setting
        args.wandb_project = 'CleanGPT'
        args.exp_name = f'Adder({args.adder_ndigit}_format)' if args.adder_use_format else f'Adder({args.adder_ndigit})'
        args.exp_profile = f'{args.exp_name}_{args.model}_{args.n_position}_{args.n_embed}_{args.n_head}_{args.n_layer}'
        args.exp_profile = f'{args.exp_profile}_compiled' if args.compile else args.exp_profile
        args.exp_profile = f'{args.exp_profile}_ampd' if args.use_amp else args.exp_profile

        return args