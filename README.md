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

## TODO
- Support training with mixed datasets
- Support llama model
- Support kvcache
- Support RLHF
- Support multimodal input
- Extend this project to control tasks similar to [Gato](https://arxiv.org/pdf/2205.06175)