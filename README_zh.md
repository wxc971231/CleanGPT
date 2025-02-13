**其他语言: [English](README.md), [中文](README_zh.md).**

# CleanGPT

CleanGPT：一个基于PyTorch实现的[GPT](https://github.com/openai/gpt-2)类模型训练框架。本项目试图保持清晰、简洁、扩展性和教育性，旨在为科研工作提供一个易于使用的工程模板。本项目基于 [NanoGPT](https://github.com/karpathy/nanoGPT) 扩展实现



## 特性
1. **分布式训练**：支持基于 PyTorch DDP 的多卡训练框架
2. **自动混合精度**：支持基于 `torch.cuda.amp` 的混合精度训练
3. **模型编译加速**：支持利用 `torch.compile` 对模型进行编译优化从而加速训练（要求 Pytorch 2.0 及以上版本）
4. **轻量数据加载**：利用 `np.memmap` 构造 Dataloader，不需要将全部数据加载到内存
5. **训练调度器**：提供了强大的训练调度器，支持 learning rate、weight decay coefficient 和训练 batch size 的动态调度，使用早停机制避免过拟合
6. **断点续训**：支持从最新的 snapshot 无感恢复训练过程
7. **模型管理**：提供了实用的 checkpoint 保存管理机制，可根据设定自动保存最好（即验证损失最低）的n个模型权重，且可从指定 checkpoint 初始化进行微调
8. **Wandb Log**：支持在 [Wandb](https://wandb.ai/site) 实时记录训练损失、验证损失、学习率、数据集访问比例等数据曲线
9. **Macro Batch**：由于 Lanuage Model 训练往往使用非常大的数集，整个训练过程可能只遍历数据集几次，甚至无法完整遍历一次，传统的 epoch 概念不再适用。本项目基于 macro-batch 概念进行训练，具体地，batch 是加载数据的最小单位，若干个 batch 组成一个 macro-batch，作为验证损失评估、snapshot & checkpoint 保存的单位
10. **GPT2**: 支持加载 HuggingFace GPT-2 checkpoints 作为初始模型进行微调

## 部署指南
1. 安装 Python 3.9 及以上版本
2. 克隆项目
    ```
    git clone https://github.com/wxc971231/CleanGPT.git
    cd CleanGPT
    ```
3. 安装 Pytorch：根据你的 CUDA 版本，在[官网](https://pytorch.org/get-started/previous-versions/)找到安装命令。推荐安装 Pytorch 2.0.1 及以上版本
4. 安装依赖
    ```
    pip install -r requirements.txt
    ```

## TODO
- 支持混合数据集训练
- 支持 llama 模型
- 支持 kvcahce
- 支持 RLHF
- 支持多模态输入
- 将本项目扩展至类似 [Gato](https://arxiv.org/pdf/2205.06175) 的控制任务 