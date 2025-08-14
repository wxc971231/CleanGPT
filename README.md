# CleanGPT

**Read in other languages: [English](README.md), [中文](README_zh.md).**

CleanGPT is a clean, educational, and extensible PyTorch-based training framework for GPT-style language models. Built upon [NanoGPT](https://github.com/karpathy/nanoGPT), this project aims to provide researchers and practitioners with a well-structured, easy-to-understand codebase for training transformer-based language models.

## ✨ Features

### 🚀 Training Infrastructure
- **Distributed Training**: Multi-GPU training support with PyTorch DDP
- **Automatic Mixed Precision**: Memory-efficient training with `torch.cuda.amp`
- **Model Compilation**: Accelerated training with `torch.compile` (PyTorch 2.0+)
- **Resume Training**: Seamless training resumption from snapshots
- **Early Stopping**: Automatic overfitting prevention

### 📊 Advanced Scheduling
- **Dynamic Learning Rate**: Cosine/linear decay with warmup
- **Weight Decay Scheduling**: Adaptive regularization
- **Gradient Accumulation**: Dynamic batch size scaling
- **Macro-Batch Training**: Efficient large-scale dataset handling

### 🔧 Model Support
- **NanoGPT**: Lightweight GPT implementation
- **Llama Architecture**: Modern transformer with RMSNorm and SwiGLU
- **Flexible Configuration**: Easy model architecture customization

### 📈 Monitoring & Management
- **Wandb Integration**: Real-time training metrics visualization
- **Checkpoint Management**: Automatic best model preservation
- **Memory-Efficient Data Loading**: `np.memmap` for large datasets
- **Comprehensive Evaluation**: Built-in scoring for mathematical tasks

## 🗂️ Project Structure

```
CleanGPT/
├── configs/                # Experiment configurations
│   ├── base.py                # Base configuration class
│   ├── shakespeare.py         # Shakespeare dataset config
│   ├── mathAdder.py           # Addition task config
│   ├── mathMulitplier.py      # Multiplication task config
│   └── tinystory.py           # TinyStory dataset config
├── data/                   # Dataset preparation and loading
│   ├── data.py                # Core data loading utilities
│   ├── shakespeare_char/      # Character-level Shakespeare
│   ├── adder/                 # Mathematical addition dataset
│   ├── multiplier/            # Mathematical multiplication dataset
│   └── tinystory/             # TinyStory dataset
├── model/                  # Model architectures
│   ├── NanoGPT.py             # GPT implementation
│   └── llama.py               # Llama architecture
├── train/                  # Training infrastructure
│   ├── trainer.py             # Main training logic
│   ├── train_ddp.py           # Distributed training entry
│   ├── scheduler.py           # Learning rate and parameter scheduling
│   └── config.py              # Command-line argument parsing
├── eval/                   # Evaluation and testing
│   ├── evaluater.py           # Model evaluation framework
│   ├── eval_ddp.py            # Distributed evaluation
│   └── script_score.py        # Task-specific scoring
└── utils/                  # Utility functions
    ├── utils.py               # General utilities
    └── utils_model.py         # Model-specific utilities
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/wxc971231/CleanGPT.git
   cd CleanGPT
   ```

2. **Install PyTorch**
   
   Install PyTorch according to your CUDA version from the [official website](https://pytorch.org/get-started/previous-versions/). PyTorch 2.0+ is recommended for model compilation features.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Training Examples

#### 1. Shakespeare Character-Level Model

```bash
# Prepare dataset
cd data/shakespeare_char
python prepare.py
cd ../..

# Train model
torchrun --standalone --nproc_per_node=1 train/train_ddp.py
```

#### 2. Mathematical Addition Task

```bash
# Prepare dataset
cd data/adder
python prepare.py
cd ../..

# Train model (modify experiment_name in train_ddp.py to 'Adder_NanoGPT')
torchrun --standalone --nproc_per_node=1 train/train_ddp.py
```

#### 3. Multi-GPU Training

```bash
# Train on 4 GPUs
torchrun --standalone --nproc_per_node=4 train/train_ddp.py
```

## 📋 Supported Datasets

| Dataset | Description | Task Type | Vocab Size |
|---------|-------------|-----------|------------|
| **Shakespeare** | Character-level text from Shakespeare works | Language Modeling | ~65 |
| **TinyStory** | Children's stories dataset | Language Modeling | ~32K |
| **Mathematical Addition** | N-digit addition problems | calculate | 10-13 |
| **Mathematical Multiplication** | N-digit multiplication | calculate | 10-13 |

## ⚙️ Configuration System

CleanGPT uses a flexible configuration system with experiment-specific settings:

```python
# Example: Shakespeare configuration
class ShakespeareNanoGPTConfig(BaseExperimentConfig):
    def get_args_ready(self):
        args = self.get_base_args()
        
        # Model settings
        args.model = 'NanoGPT'
        args.n_position = 256
        args.n_layer = 4
        args.n_head = 8
        args.n_embed = 256
        
        # Training settings
        args.lr_max = 1e-3
        args.batch_size_per_gpu = 32
        args.train_iters = 5000
        
        return args
```

## 📊 Monitoring and Evaluation

### Wandb Integration

Training metrics are automatically logged to Wandb:
- Training/Validation loss
- Learning rate schedules
- Model parameters
- Dataset coverage

### Mathematical Task Evaluation

For addition and multiplication tasks, the framework includes specialized evaluation:

```python
# Automatic accuracy calculation
accuracy = eval_score_adder(model, dataset, tokenizer)
print(f"Addition accuracy: {accuracy:.2%}")
```

## 🔧 Advanced Features

### Dynamic Parameter Scheduling

```python
# Learning rate with warmup and cosine decay
args.lr_warmup_ratio = 0.05
args.lr_decay_ratio = 0.95
args.lr_decay_style = "cosine"

# Dynamic gradient accumulation
args.ga_begin = 4
args.ga_end = 16
args.ga_incr_style = "linear"
```

### Memory-Efficient Data Loading

```python
# Large datasets loaded with memory mapping
data = np.load(data_path, mmap_mode='r')
# No need to load entire dataset into memory
```

### Model Compilation (PyTorch 2.0+)

```python
# Automatic model compilation for speedup
if args.compile:
    model = torch.compile(model)
```

## 📈 Performance Tips

1. **Use Mixed Precision**: Enable `--use-amp` for memory efficiency
2. **Optimize Batch Size**: Use gradient accumulation for effective large batches
3. **Enable Compilation**: Use `--compile` with PyTorch 2.0+
4. **Multi-GPU Training**: Scale to multiple GPUs with DDP
5. **Memory Mapping**: Efficient for datasets larger than RAM