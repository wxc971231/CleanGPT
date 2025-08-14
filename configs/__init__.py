from configs.shakespeare import *
from configs.tinystory import *
from configs.mathAdder import *
from configs.mathMulitplier import *
from configs.base import BaseExperimentConfig

# 实验注册表
EXPERIMENT_REGISTRY = {
    'Shakespeare_NanoGPT': ShakespeareNanoGPTConfig,
    'TinyStory_NanoGPT': TinyStoryNanoGPTConfig,
    'TinyStory_Llama': TinyStoryLlamaConfig,
    'Adder_NanoGPT': AdderNanoGPTConfig,
    'Adder_Llama': AdderLlamaConfig,
    'Multiplier_NanoGPT': MultiplierNanoGPTConfig,
    'Multiplier_Llama': MultiplierLlamaConfig,
}

def get_experiment_config(experiment_name: str) -> BaseExperimentConfig:
    """获取实验配置"""
    if experiment_name not in EXPERIMENT_REGISTRY:
        available = list(EXPERIMENT_REGISTRY.keys())
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {available}")
    
    return EXPERIMENT_REGISTRY[experiment_name]().get_args_ready()