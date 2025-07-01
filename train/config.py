from argparse import ArgumentParser

def str2bool(x):
    assert x == "True" or x == "False"
    return True if x == "True" else False

def parse_args():
    parser = get_parser_for_basic_args()
    parser = _add_training_args(parser)
    parser = _add_model_args(parser)
    parser = _add_scheduler_args(parser)
    args = parser.parse_args()
    return args

def get_parser_for_basic_args():
    parser = ArgumentParser("Basic Configuration")
    parser.add_argument(
        "--seeds", type=int, nargs='+', default=[42, 43, 44], 
        help="Random seed for numpy/torch"
    )
    parser.add_argument(
        "--out-dir", type=str, default="out",
        help="The path of output, where the checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--dataset", type=str, default="shakespeare_char",
        help="The dataset name, which is a floder",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, 
        help="How many subprocesses to use for data loading in torch.utils.data.Dataloader"
    )
    parser.add_argument(
        "--exp-name", type=str, default=None,
        help="The short brief name of this experiment, which will be used for the group name on wandb log.",
    )
    parser.add_argument(
        "--exp-profile", type=str, default=None,
        help="The name of this experiment with more details, which will be used to construct the ckpt storage path.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="shakespeare_char",
        help="The project name on wandb",
    )
    parser.add_argument(
        "--wandb", type=str2bool, default=False, nargs="?", const=True,     
        help="Synchronize the experiment curve to wandb",
    )
    parser.add_argument(
        "--save-ckpt", type=str2bool, default=True, nargs="?", const=True,     
        help="Save model checkpoint during training",
    )
    parser.add_argument(
        "--save-ckpt-num", type=int, default=0, 
        help="The number of checkpoints to save, 0 for saving all checkpoints",
    )
    parser.add_argument(
        "--save-snapshot", type=str2bool, default=True, nargs="?", const=True,     
        help="Save model snapshot during training",
    )
    parser.add_argument(
        "--save-strategy", type=str, default="best",
        choices=["best", "interval"], 
        help="Checkpoint saving strategy.",
    )
    parser.add_argument(
        "--use-kvcache", type=str2bool, default=True, nargs="?", const=True,     
        help="Use kvcahce to speed up froward pass",
    )
    parser.add_argument(
        "--compile", type=str2bool, default=False, nargs="?", const=True,     
        help="use PyTorch 2.0 to compile the model to be faster",
    )
    return parser

def _add_model_args(parser):
    group = parser.add_argument_group(title="model")
    group.add_argument(
        "--model", type=str, default="NanoGPT",
        choices=["NanoGPT", "llama"], 
        help="Choose the language model to use.",
    )
    group.add_argument(
        "--init-from", type=str, default=None,
        help='''
            'None' for training from scratch or resuming from latest snapshot within out-dir.
            '*.pt' for finetuning from pretrained ckpt '*.pt'
            'gpt2*' for finetuning from huggingface GPT2 ckpt, e.g. 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        '''
    )
    group.add_argument(
        "--vocab-size", type=int, default=50304,
        help="GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency",
    )
    group.add_argument(
        "--n-embed", type=int, default=768,
        help="width of the embedding vectors.",
    )
    group.add_argument(
        "--n-position", type=int, default=1024,
        help="The maximum sequence length that this model might ever be used with."
        "Typically set this to something large just in case (e.g., 512 or 1024 or 2048)",
    )
    group.add_argument(
        "--n-layer", type=int, default=12,
        help="Number of hidden layers in the Transformer encoder",
    )
    group.add_argument(
        "--n-head", type=int, default=12,
        help="Number of attention heads for each attention layer in the Transformer encoder.",
    )
    group.add_argument(
        "--n-inner", type=int, default=None,
        help="Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embed",
    )
    group.add_argument(
        "--activation-fn", type=str, default="gelu",
        choices=['relu', 'gelu', 'tanh', 'sigmoid', 'geglu'],
        help="Activation function",
    )
    group.add_argument(
        "--layer-norm-epsilon", type=float, default=1e-5,
        help="The epsilon to use in the layer normalization layers.",
    )
    group.add_argument(
        "--rms-norm-eps", type=float, default=1e-6,
        help="RMS Norm epsilon, special for llama.",
    )
    group.add_argument(
        "--dropout", type=float, default=0.2,
        help="The general dropout probability.",
    )
    group.add_argument(
        "--weight-tying", type=str2bool, default=True,
        help="Whether to share embedding weights between input token-embedding layer and output lm-head layer"
        "https://paperswithcode.com/method/weight-tying"
    )
    group.add_argument(
        "--add-bias", type=str2bool, default=False,
        help="Do we use bias inside LayerNorm and Linear layers."
        "True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster"
    )
    group.add_argument(
        "--use-amp", type=str2bool, default=False,
        help="Whether to use automatic mixed precision (AMP) to speed up training.",
    )

    return parser

def _add_training_args(parser):
    group = parser.add_argument_group(title="training")

    # Early stopping setting
    group.add_argument(
        "--use-early-stopping", type=str2bool, default=False,
        help="Whether to use the early stopping mechanism"
    )
    group.add_argument(
        "--early-stopping-patience", type=int, default=7,
        help="How long to wait after last time validation loss improved."
    )
    group.add_argument(
        "--early-stopping-delta", type=int, default=0,
        help="Minimum change in the monitored quantity to qualify as an improvement."
    )
    
    # Training setting
    group.add_argument(
        "--train-iters", type=int, default=5000,
        help="Total number of iterations(batch) to train over all training runs."
    )
    group.add_argument(
        "--batch-size-per-gpu", type=int, default=64,
        help="Training batch size (per GPU)."
    )
    group.add_argument(
        "--batch-grad-accum-step", type=int, default=1, 
        help="Batch step of Gradient Accumulation."
    )
    group.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size, equals to (batch_size_per_gpu * world_size * batch_grad_accum_step)."
    )
    
    group.add_argument(
        "--eval-batch-num", type=int, default=500,
        help="Batch num for each evaluation during training."
    )
    group.add_argument(
        "--eval-batch-size-per-gpu", type=int, default=64,
        help="Evaluating batch size (per GPU)"
    )
    group.add_argument(
        "--eval-batch-size", type=int, default=64,
        help="Evaluating batch size, equals to (eval_batch_size_per_gpu * world_size)."
    )
    
    group.add_argument(
        "--snapshot-save-interval", type=int, default=5,
        help="Batch interval between two snapshots saved"
    )
    group.add_argument(
        "--eval-interval", type=int, default=5,
        help="Batch interval for eval loss calculating"
    )
    group.add_argument(
        "--log-interval", type=int, default=5,
        help="Batch interval for logging"
    )
    group.add_argument(
        "--skip-first-eval", type=str2bool, default=True,
        help="Whether to skip the first evaluation at batch 0",
    )
    

    # Optimizer setting
    group.add_argument(
        "--clip-grad", type=float, default=1.0,
        help="clip gradients at this value, set clip-grad == 0.0 to disable clipping.",
    )
    group.add_argument(
        "--adam-beta1", type=float, default=0.9,
        help="First coefficient for computing running averages of gradient and its square",
    )
    group.add_argument(
        "--adam-beta2", type=float, default=0.95,
        help="Second coefficient for computing running averages of gradient and its square",
    )
    group.add_argument(
        "--adam-eps", type=float, default=1e-08,
        help="Term added to the denominator to improve" "numerical stability",
    )
    
    return parser

def _add_scheduler_args(parser):
    group = parser.add_argument_group(title="scheduler")
    group.add_argument(
        "--override-opt-param-scheduler", type=str2bool, default=False, nargs="?", const=False,     
        help="Set 'True' to override all scheduler setting, otherwise the scheduler setting will be set by checkpoint.",
    )

    # learning rate setting
    group.add_argument(
        "--lr-decay-style", type=str, default="cosine", 
        choices=["constant", "linear", "cosine"],
        help="Learning rate decay function.",
    )
    group.add_argument(
        "--lr-decay-factor", type=float, default=10,
        help="The final learning rate equals to (lr_max/lr_decay_factor)"
        "With baby networks it's OK to use 'constant', but for large networks usually decrease to 10%"
    )
    group.add_argument(
        "--lr-warmup-ratio", type=float, default=0.1,
        help="The proportion of the number of warmup iterations to the total number of iterations",
    )
    group.add_argument(
        "--lr-decay-ratio", type=float, default=0.8,
        help="the proportion of the number of decay iterations to the total number of iterations",
    )
    group.add_argument(
        "--lr-begin", type=float, default=0.0,
        help="The initial learning rate."
    )
    group.add_argument(
        "--lr-max", type=float, default=5e-5,
        help="The learning rate peak at end of warmup process. It's as same as lr-begin if we use constant lr",
    )

    # weight decay setting
    group.add_argument(
        "--wd-decr-style", type=str, default="constant",
        choices=["constant", "linear", "cosine"],
        help="Weight decay coefficient decrease function."
        "For most of situation, keep the weight decay coefficient 'constant' is suitable, decreasing it can sometimes help to improve the performance."
    )
    group.add_argument(
        "--wd-begin", type=float, default=1e-2,
        help="The initial weight decay coefficient for L2 regularization."
        "The baby networks can afford to go a bit higher (1e-4 ~ 1e-2), for large networks usually (1e-5 ~ 1e-3)"
    )
    group.add_argument(
        "--wd-end", type=float, default=1e-2,
        help="The final weight decay coefficient for L2 regularization.",
    )

    # grad accum step setting
    group.add_argument(
        "--grad-accum-step-incr-style", type=str, default="constant", 
        choices=["constant", "linear", "power"],
        help="grad accum step incr function."
        "With baby networks we can simply use 'constant' grad_accum_step, but for large networks sometimes increase to 2x~10x"
    )
    group.add_argument(
        "--ga-begin", type=int, default=1,
        help="The initial grad accum step",
    )
    group.add_argument(
        "--ga-end", type=int, default=10,
        help="The final grad accum step",
    )

    return parser