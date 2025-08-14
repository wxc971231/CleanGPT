# CleanGPT

**其他语言版本: [English](README.md), [中文](README_zh.md).**

CleanGPT 是一个基于 PyTorch 的清晰、教育性和可扩展的 GPT 风格语言模型训练框架。本项目基于 [NanoGPT](https://github.com/karpathy/nanoGPT) 构建，旨在为研究人员和从业者提供一个结构良好、易于理解的代码库，用于训练基于 Transformer 的语言模型。

## ✨ 特性

### 🚀 训练基础设施
- **分布式训练**: 基于 PyTorch DDP 的多 GPU 训练支持
- **自动混合精度**: 使用 `torch.cuda.amp` 进行内存高效训练
- **模型编译**: 使用 `torch.compile` 加速训练 (PyTorch 2.0+)
- **断点续训**: 从快照无缝恢复训练
- **早停机制**: 自动防止过拟合

### 📊 高级调度
- **动态学习率**: 余弦/线性衰减与预热
- **权重衰减调度**: 自适应正则化
- **梯度累积**: 动态批次大小缩放
- **宏批次训练**: 高效的大规模数据集处理

### 🔧 模型支持
- **NanoGPT**: 轻量级 GPT 实现
- **Llama 架构**: 使用 RMSNorm 和 SwiGLU 的现代 Transformer
- **灵活配置**: 简单的模型架构自定义

### 📈 监控与管理
- **Wandb 集成**: 实时训练指标可视化
- **检查点管理**: 自动保存最佳模型
- **内存高效数据加载**: 使用 `np.memmap` 处理大型数据集
- **全面评估**: 数学任务的内置评分

## 🗂️ 项目结构

```
CleanGPT/
├── configs/                # 实验配置
│   ├── base.py                # 基础配置类
│   ├── shakespeare.py         # 莎士比亚数据集配置
│   ├── mathAdder.py           # 加法任务配置
│   ├── mathMulitplier.py      # 乘法任务配置
│   └── tinystory.py           # TinyStory 数据集配置
├── data/                   # 数据集准备和加载
│   ├── data.py                # 核心数据加载工具
│   ├── shakespeare_char/      # 字符级莎士比亚数据
│   ├── adder/                 # 数学加法数据集
│   ├── multiplier/            # 数学乘法数据集
│   └── tinystory/             # TinyStory 数据集
├── model/                  # 模型架构
│   ├── NanoGPT.py             # GPT 实现
│   └── llama.py               # Llama 架构
├── train/                  # 训练基础设施
│   ├── trainer.py             # 主要训练逻辑
│   ├── train_ddp.py           # 分布式训练入口
│   ├── scheduler.py           # 学习率和参数调度
│   └── config.py              # 命令行参数解析
├── eval/                   # 评估和测试
│   ├── evaluater.py           # 模型评估框架
│   ├── eval_ddp.py            # 分布式评估
│   └── script_score.py        # 任务特定评分
└── utils/                  # 工具函数
    ├── utils.py               # 通用工具
    └── utils_model.py         # 模型特定工具
```

## 🚀 快速开始

### 安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/wxc971231/CleanGPT.git
   cd CleanGPT
   ```

2. **安装 PyTorch**
   
   根据您的 CUDA 版本从[官方网站](https://pytorch.org/get-started/previous-versions/)安装 PyTorch。推荐使用 PyTorch 2.0+ 以获得模型编译功能。

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 训练示例

#### 1. 莎士比亚字符级模型

```bash
# 准备数据集
cd data/shakespeare_char
python prepare.py
cd ../..

# 训练模型
torchrun --standalone --nproc_per_node=1 train/train_ddp.py
```

#### 2. 数学加法任务

```bash
# 准备数据集
cd data/adder
python prepare.py
cd ../..

# 训练模型 (在 train_ddp.py 中修改 experiment_name 为 'Adder_NanoGPT')
torchrun --standalone --nproc_per_node=1 train/train_ddp.py
```

#### 3. 多 GPU 训练

```bash
# 在 4 个 GPU 上训练
torchrun --standalone --nproc_per_node=4 train/train_ddp.py
```

## 📋 支持的数据集

| 数据集 | 描述 | 任务类型 | 词汇表大小 |
|--------|------|----------|------------|
| **Shakespeare** | 莎士比亚作品的字符级文本 | 语言建模 | ~65 |
| **TinyStory** | 儿童故事数据集 | 语言建模 | ~32K |
| **数学加法** | N 位数加法问题 | 数学计算 | 10-13 |
| **数学乘法** | N 位数乘法问题 | 数学计算 | 10-13 |

## ⚙️ 配置系统

CleanGPT 使用灵活的配置系统，支持实验特定设置：

```python
# 示例：莎士比亚配置
class ShakespeareNanoGPTConfig(BaseExperimentConfig):
    def get_args_ready(self):
        args = self.get_base_args()
        
        # 模型设置
        args.model = 'NanoGPT'
        args.n_position = 256
        args.n_layer = 4
        args.n_head = 8
        args.n_embed = 256
        
        # 训练设置
        args.lr_max = 1e-3
        args.batch_size_per_gpu = 32
        args.train_iters = 5000
        
        return args
```

## 📊 监控和评估

### Wandb 集成

训练指标自动记录到 Wandb：
- 训练/验证损失
- 学习率调度
- 模型参数
- 数据集覆盖率

### 数学任务评估

对于加法和乘法任务，框架包含专门的评估：

```python
# 自动准确率计算
accuracy = eval_score_adder(model, dataset, tokenizer)
print(f"加法准确率: {accuracy:.2%}")
```

## 🔧 高级特性

### 动态参数调度

```python
# 带预热和余弦衰减的学习率
args.lr_warmup_ratio = 0.05
args.lr_decay_ratio = 0.95
args.lr_decay_style = "cosine"

# 动态梯度累积
args.ga_begin = 4
args.ga_end = 16
args.ga_incr_style = "linear"
```

### 内存高效数据加载

```python
# 使用内存映射加载大型数据集
data = np.load(data_path, mmap_mode='r')
# 无需将整个数据集加载到内存
```

### 模型编译 (PyTorch 2.0+)

```python
# 自动模型编译以加速
if args.compile:
    model = torch.compile(model)
```

## 📈 性能优化建议

1. **使用混合精度**: 启用 `--use-amp` 提高内存效率
2. **优化批次大小**: 使用梯度累积实现有效的大批次
3. **启用编译**: 在 PyTorch 2.0+ 中使用 `--compile`
4. **多 GPU 训练**: 使用 DDP 扩展到多个 GPU
5. **内存映射**: 对于大于 RAM 的数据集很高效