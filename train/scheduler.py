"""Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/optimizer_param_scheduler.py
learning-rate, weight-decay and grad-accum-step scheduler"""

import math
import os

class EarlyStopping:
    """ Early stops the training if validation loss doesn't improve after a given patience. """
    def __init__(self, patience:int=6, delta:float=0.0):
        self.patience = patience    # How long to wait after last time validation loss improved.
        self.delta = delta          # Minimum change in the monitored quantity to qualify as an improvement.
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.min_val_loss - self.delta:
            self.min_val_loss = val_loss
            self.counter = 0        # Reset counter if validation loss improves            
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

class OptimizerParamScheduler(object):
    """ Anneals learning-rate, weight-decay and grad-accum-step. """
    def __init__(self, args, optimizer):
        self.rank = int(os.environ.get("RANK", default='0'))
        self.args = args
        self.optimizer = optimizer
        self.num_steps = 0
        self.total_steps = args.train_iters

        # learning rate parameters
        self.lr_decay_style = args.lr_decay_style
        self.lr_begin = float(args.lr_begin)
        self.lr_decay_factor = float(args.lr_decay_factor)
        self.lr_max = float(args.lr_max)
        self.lr_min = self.lr_max/self.lr_decay_factor
        assert self.lr_decay_factor >= 1.0
        assert 0.0 <= self.lr_min <= self.lr_max
        self.lr_warmup_steps = int(self.total_steps * args.lr_warmup_ratio)
        self.lr_decay_steps = int(self.total_steps * args.lr_decay_ratio)
        assert self.lr_decay_steps >= 0
        assert self.lr_warmup_steps + self.lr_decay_steps <= self.total_steps

        # weight decay parameters
        self.wd_decr_style = args.wd_decr_style
        self.wd_begin = float(args.wd_begin)
        self.wd_end = float(args.wd_end)
        assert 0.0 <= self.wd_end <= self.wd_begin 
        self.wd_decr_steps = self.total_steps
        
        # grad accum step parameters
        self.grad_accum_step_incr_style = args.grad_accum_step_incr_style
        self.ga_begin = int(args.ga_begin)
        self.ga_end = int(args.ga_end)
        assert 0.0 <= self.ga_begin <= self.ga_end
        self.ga_incr_steps = self.total_steps

    def get_wd(self):
        """Weight decay decr functions"""
        if self.wd_decr_style == "constant":
            assert self.wd_begin == self.wd_end
            return self.wd_end

        decay_ratio = float(self.num_steps) / float(self.wd_decr_steps)
        assert 0.0 <= decay_ratio <= 1.0

        if self.wd_decr_style == "linear":
            coeff = decay_ratio
        elif self.wd_decr_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * (1 - decay_ratio)) + 1.0)
        else:
            raise Exception(f"{self.wd_decr_style} weight decay increment style is not supported.")

        return self.wd_begin + coeff * (self.wd_end - self.wd_begin)
    
    def get_lr(self):
        """Learning rate decay functions adapted from https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""
        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.lr_begin + (self.lr_max - self.lr_begin) * float(self.num_steps) / float(self.lr_warmup_steps)

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == "constant":
            assert self.lr_max == self.lr_min == self.lr_begin
            return self.lr_max

        # For any steps larger than `self.lr_decay_steps`, use `self.lr_min`.
        if self.num_steps > self.lr_warmup_steps + self.lr_decay_steps:
            return self.lr_min

        # If we are done with the warmup period, use the decay style.
        decay_ratio = float(self.num_steps - self.lr_warmup_steps) / float(self.lr_decay_steps)
        assert 0.0 <= decay_ratio <= 1.0
        delta_lr = self.lr_max - self.lr_min

        if self.lr_decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception(f"{self.lr_decay_style} decay style is not supported.")

        return self.lr_min + coeff * delta_lr

    def get_ga_step(self):
        """grad_accum_step incr functions"""
        if self.grad_accum_step_incr_style == "constant":
            assert self.ga_begin == self.ga_end
            return self.ga_end

        incr_ratio = float(self.num_steps) / float(self.ga_incr_steps)
        assert 0.0 <= incr_ratio <= 1.0

        if self.grad_accum_step_incr_style == "linear":
            coeff = incr_ratio
        elif self.grad_accum_step_incr_style == "power":
            coeff = 1 - math.pow(0.1, incr_ratio)
        else:
            raise Exception(f"{self.grad_accum_step_incr_style} grad accum step increment style is not supported.")
        
        return int(self.ga_begin + coeff * (self.ga_end - self.ga_begin))
    
    def step(self, increment=1):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_lr = self.get_lr()
        new_ga_step = self.get_ga_step()
        new_wd = self.get_wd()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
            if group["weight_decay"] != 0:
                group["weight_decay"] = new_wd
        return new_lr, new_ga_step, new_wd

    def state_dict(self):
        state_dict = {
            "total_steps": self.total_steps,
            "num_steps": self.num_steps,
            "lr_begin": self.lr_begin,
            "lr_max": self.lr_max,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_decay_factor": self.lr_decay_factor,
            "lr_decay_style": self.lr_decay_style,
            "wd_begin": self.wd_begin,
            "wd_end": self.wd_end,
            "wd_decr_steps": self.wd_decr_steps,
            "wd_decr_style": self.wd_decr_style,
            "ga_begin": self.ga_begin,
            "ga_end": self.ga_end,
            "ga_incr_steps": self.ga_incr_steps,
            "grad_accum_step_incr_style": self.grad_accum_step_incr_style,
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and setting them."""
        if self.args.override_opt_param_scheduler:
            if self.rank == 0 and sd_value != cls_value:
                print(f" > overriding {name} value form {sd_value} to {cls_value}")
            return cls_value
        else:
            if self.rank == 0 and cls_value != sd_value:
                print(f" > using checkpoint value {sd_value} for {name}, which is not match to current setting {cls_value}")
        
        return sd_value

    def load_state_dict(self, sd):
        # keep the training-process-ratio consistent in case of the total_steps changed
        num_steps = int(sd["num_steps"] * (self.total_steps/sd["total_steps"]))
        self.step(increment=num_steps)
        if self.rank == 0 and num_steps != sd["num_steps"]:
            print(f" > set num_steps value form {sd['num_steps']} to {num_steps}")

        # lr paras
        self.lr_decay_factor = self._check_and_set(self.lr_decay_factor, sd["lr_decay_factor"], "lr decay factor")
        self.lr_begin = self._check_and_set(self.lr_begin, sd["lr_begin"], "initial learning rate")
        self.lr_max = self._check_and_set(self.lr_max, sd["lr_max"], "maximum learning rate")
        self.lr_min = self.lr_max/self.lr_decay_factor
        self.lr_warmup_steps = self._check_and_set(self.lr_warmup_steps, sd["lr_warmup_steps"], "warmup iterations")
        self.lr_decay_steps = self._check_and_set(self.lr_decay_steps, sd["lr_decay_steps"], "number of lr_decay iterations")
        self.lr_decay_style = self._check_and_set(self.lr_decay_style, sd["lr_decay_style"], "learning rate decay style")
        
        # wd paras
        if "wd_begin" in sd:
            self.wd_begin = self._check_and_set(self.wd_begin, sd["wd_begin"], "initial weight decay coefficient")
            self.wd_end = self._check_and_set(self.wd_end, sd["wd_end"], "final weight decay coefficient")
            self.wd_decr_steps = self._check_and_set(self.wd_decr_steps, sd["wd_decr_steps"], "total number of weight decay iterations",)
            self.wd_decr_style = self._check_and_set(self.wd_decr_style, sd["wd_decr_style"], "weight decay incr style")

        # grad accum para
        if "ga_begin" in sd:
            self.ga_begin = self._check_and_set(self.ga_begin, sd["ga_begin"], "initial grad accum step")
            self.ga_end = self._check_and_set(self.ga_end, sd["ga_end"], "final grad accum step")
            self.ga_incr_steps = self._check_and_set(self.ga_incr_steps, sd["ga_incr_steps"], "total number of grad incr iterations",)
            self.grad_accum_step_incr_style = self._check_and_set(self.grad_accum_step_incr_style, sd["grad_accum_step_incr_style"], "grad accum step incr style")

        return self.total_steps/sd["total_steps"]
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    class Args:
        def __init__(self):
            self.train_iters = 2000
            self.lr_begin = 0.0
            self.lr_decay_factor = 10
            self.lr_max = 0.01
            self.lr_warmup_ratio = 0.1
            self.lr_decay_ratio = 0.75
            self.lr_decay_style = 'cosine'
            self.wd_decr_style = 'cosine'
            self.wd_begin = 0.01
            self.wd_end = 0.001
            self.grad_accum_step_incr_style = 'power'
            self.ga_begin = 1
            self.ga_end = 20

    # 模拟的优化器参数
    class MockOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': 0.0, 'weight_decay': 0.0}]

    # 创建模拟参数和优化器
    args = Args()
    optimizer = MockOptimizer()
    scheduler = OptimizerParamScheduler(args, optimizer)    # 创建调度器

    # 模拟训练过程
    lrs, wds, gas = [],  [],  []
    for step in range(args.train_iters):
        scheduler.num_steps = step
        new_lr, new_ga_step, new_wd = scheduler.step()
        lrs.append(new_lr)
        wds.append(new_wd)
        gas.append(new_ga_step)

    # 绘制折线图
    fig, axs = plt.subplots(3, 1, figsize=(5, 5))

    # 学习率变化曲线
    axs[0].plot(lrs, label='Learning Rate')
    axs[0].set_title('Learning Rate Schedule')
    axs[0].set_xlabel('Training Steps')
    axs[0].set_ylabel('Learning Rate')
    axs[0].legend()

    # 权重衰减变化曲线
    axs[1].plot(wds, label='Weight Decay', color='orange')
    axs[1].set_title('Weight Decay Schedule')
    axs[1].set_xlabel('Training Steps')
    axs[1].set_ylabel('Weight Decay')
    axs[1].legend()

    # 梯度累积步数变化曲线
    axs[2].plot(gas, label='Grad Accum Steps', color='green')
    axs[2].set_title('Grad Accum Steps Schedule')
    axs[2].set_xlabel('Training Steps')
    axs[2].set_ylabel('Grad Accum Steps')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
