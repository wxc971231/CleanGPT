**Read in other languages: [English](README.md), [中文](README_zh.md).**

# CleanGPT

CleanGPT: A training framework for GPT-style model implemented with PyTorch. CleanGPT tries to be clear, concise, extensible, and educational, serving as an easy-to-use template for research purposes. The project is an extension built upon [NanoGPT](https://github.com/karpathy/nanoGPT).

## Features
1. **Distributed Training**: Supports multi-GPU training framework based on PyTorch DDP.
2. **Automatic Mixed Precision**: Supports mixed-precision training using `torch.cuda.amp`.
3. **Model Compilation Acceleration**: Supports model compilation optimization with `torch.compile` to accelerate training (requires PyTorch 2.0 or above).
4. **Lightweight Data Loading**: Constructs DataLoader using `np.memmap`, which eliminates the need to load the entire dataset into memory.
5. **Training Scheduler**: Provides a powerful training scheduler that supports dynamic scheduling of learning-rate, weight-decay-coefficient and training batch-size, using early stopping to prevent overfitting.
6. **Resume Training**: Supports seamless resumption of training from the latest snapshot.
7. **Ckpt Management**: Offers a practical checkpoint management mechanism that automatically saves the best _n_ model weights (i.e., with the lowest validation loss) based on user settings, and supports initialization for fine-tuning from a specified checkpoint.
8. **Wandb Logging**: Supports real-time logging of training-loss, validation-loss, learning-rate, dataset-visited-ratios and more on [Wandb](https://wandb.ai/site).
9. **Macro Batch**: As language model training typically involves extremely large datasets, the entire training process may only traverse the dataset a few times or not even complete one full pass. The traditional concept of "epoch" becomes unsuitable. In this project, the training is based on the concept of "macro-batch". Specifically, a "batch" is the smallest unit for loading data, several batches form a macro-batch, which serves as the unit for validation loss evaluation, snapshot & checkpoint saving.
10. **Init from GPT2**: Supports loading HuggingFace GPT-2 checkpoints as the initial model for fine-tuning.

## Deployment Guide
1. Install Python 3.9 or above.
2. Clone the project:
    ```
    git clone https://github.com/wxc971231/CleanGPT.git
    cd CleanGPT
    ```
3. Install PyTorch: According to your CUDA version, find the appropriate installation command from the [official website](https://pytorch.org/get-started/previous-versions/). It is recommended to install PyTorch 2.0.1 or above.
4. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Training Example
1. Build the dataset
    ```shell
    cd data/shakespeare_char
    python prepare.py
    ```
2. Set hyperparameters in the `get_args_ready` method in `train/train_ddp.py`, like:
    ```python
    def get_args_ready(WORLD_SIZE:int, RANK:int):
        args = parse_args()
        args.world_size = WORLD_SIZE

        # model setting
        args.model = 'NanoGPT'
        args.n_position = 1024
        args.n_layer = 6
        args.n_head = 4
        args.n_embed = 384
        args.n_inner = 4 * args.n_embed
        args.dropout = 0.0                          
        args.init_from = None                       

        # optimizer setting
        args.lr_begin = 0                                       
        args.lr_max = 1e-3                          
        args.lr_decay_factor = 10.0                 
        args.lr_warmup_ratio = 0.05
        args.lr_decay_ratio = 0.95
        args.lr_decay_style = "cosine"
        args.wd_begin = 1e-3                        
        args.wd_end = args.wd_begin                 
        args.wd_decr_style = "constant"            
        args.ga_begin = 2                           
        args.ga_end = args.ga_begin                 
        args.grad_accum_step_incr_style = "constant"
        args.adam_beta2 = 0.99                      
        ...
    ```
    Detailed explanations of all hyperparameters can be found in `train/config.py`. Compared to passing parameters via command line, explicitly fixing training hyperparameters in this way is clearer and ensures reproducibility by saving the training script.

3. Start training, currently only supports single-node multi-GPU parallelism. Checkpoints & Snapshots will be saved in the path `out`.
    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=gpu ./train/train_ddp.py 
    ```
    Typically, we train until overfitting and early stopping is triggered.
    ![](img/train_log.png)
4. Evaluate the best Checkpoint. Paste the output file path generated during training into `text_autoregress.py`, it will automatically load the best Checkpoint for autoregressive generation. An example trained on the TinyStory dataset is as follows:
    ```text
    Once upon a time, 3 year old girl named Lucy wanted to go on an adventure. She asked her mom if she could go. Her mom said yes, but only if she stayed close.
    Lucy was so excited! She ran outside and started exploring. She found a big tree and decided to climb it. She climbed higher and higher until she was at the top.
    At the top, Lucy looked around and saw a beautiful view. She wanted to stay and enjoy it for a while. Then she carefully climbed down and ran back home.
    When she got home, her mom asked her what she was doing. Lucy said, "I'm going on an adventure!" Her mom smiled and said, "That sounds like a great idea!"
    Lucy was so happy. She had a wonderful time exploring and eating the view. She was so glad she had stayed close to home.</s>
    ```
## TODO

| Item  | Note  |
|-------|-----|
| Support training with mixed datasets   | -  |
| Support llama model | Done (hugging face llama)  | 
| Support kvcache | Done (for llama) |
| Support RLHF | - |
|Support multimodal input|-|
|Extend this project to control tasks similar to [Gato](https://arxiv.org/pdf/2205.06175)|-|