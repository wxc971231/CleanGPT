import math
import os
from dataclasses import dataclass
import numpy as np
from utils.utils import clean_print

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
            local_rank = int(os.environ.get("LOCAL_RANK", default='0'))
            clean_print(f'EarlyStopping counter: {self.counter} out of {self.patience}', local_rank, '[Trainer]')
            if self.counter >= self.patience:
                self.early_stop = True

@dataclass
class ScheduleConfig:
    begin: float = 0.0                  # inital value
    middle: float = 0.0                 # warmup to middle value from initial value
    end: float = 0.0                    # schedule to final value from middle value
    data_type: str = 'float'            # 'float' or 'int'
    schedule_style: str = 'constant'
    schedule_factor: float = 1.0
    schedule_ratio: float = 1.0
    warmup_ratio: float = 0.0
    
class ParamScheduler(object):
    def __init__(self, config:ScheduleConfig, total_steps:int):
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0

        assert self.config.schedule_factor >= 0.0
        assert self.config.warmup_ratio + self.config.schedule_ratio <= 1.0
        assert self.config.schedule_style in ['constant', 'linear', 'cosine', 'power']
        if self.config.schedule_style == 'constant':
            assert self.config.begin == self.config.end == self.config.middle
        else:
            self.config.end = self.config.middle * self.config.schedule_factor

        self.warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        self.schedule_steps = int(self.total_steps * self.config.schedule_ratio)

    def step(self, increment=1):
        self.current_step += increment

        # Use linear warmup for the initial part.
        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            return self.config.begin + (self.config.middle - self.config.begin) * float(self.current_step) / float(self.warmup_steps)

        # If the learning rate is constant, just return the initial value.
        if self.config.schedule_style == "constant":
            return self.config.begin

        # For any steps larger than `self.warmup_steps + self.schedule_steps`, use the final value.
        if self.current_step > self.warmup_steps + self.schedule_steps:
            return self.config.end

        # If we are done with the warmup period, use the schedule_style.
        ratio = float(self.current_step - self.warmup_steps) / float(self.schedule_steps)
        assert 0.0 <= ratio <= 1.0
        delta = self.config.middle - self.config.end

        if self.config.schedule_style == "linear":
            coeff = 1.0 - ratio
        elif self.config.schedule_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * ratio) + 1.0)
        elif self.config.schedule_style == "power":
            coeff = math.pow(0.1, ratio)
        else:
            raise Exception(f"{self.config.schedule_style} schedule_style is not supported.")

        value = self.config.end + coeff * delta
        if self.config.data_type == 'int':
            value = int(value)
        return value
    
class OptimizerParamScheduler(object):
    """ Anneals learning-rate, weight-decay and grad-accum-step. """
    def __init__(self, args, optimizer, is_critic=False):
        self.rank = int(os.environ.get("RANK", default='0'))
        self.args = args
        self.optimizer = optimizer
        self.num_steps = 0

        # learning rate parameters
        self.lr_decay_style = args.lr_decay_style if not is_critic else args.lr_decay_style_critic
        self.lr_begin = float(args.lr_begin) if not is_critic else float(args.lr_begin_critic)
        lr_decay_factor = float(args.lr_decay_factor) if not is_critic else float(args.lr_decay_factor_critic)
        self.lr_max = float(args.lr_max) if not is_critic else float(args.lr_max_critic)
        self.lr_min = self.lr_max/lr_decay_factor
        assert lr_decay_factor >= 1.0
        assert 0.0 <= self.lr_min <= self.lr_max
        self.lr_decay_factor = 1.0/lr_decay_factor
        lr_warmup_ratio = args.lr_warmup_ratio if not is_critic else args.lr_warmup_ratio_critic
        lr_decay_ratio = args.lr_decay_ratio if not is_critic else args.lr_decay_ratio_critic
        self.lr_warmup_steps = int(args.train_iters * lr_warmup_ratio)
        self.lr_decay_steps = int(args.train_iters * lr_decay_ratio)
        assert self.lr_decay_steps >= 0
        assert self.lr_warmup_steps + self.lr_decay_steps <= args.train_iters
        lr_schedule_config = ScheduleConfig(
            begin=self.lr_begin,
            middle=self.lr_max,
            end=self.lr_min,
            schedule_style=self.lr_decay_style,
            schedule_factor=self.lr_decay_factor,
            warmup_ratio=lr_warmup_ratio,
            schedule_ratio=lr_decay_ratio
        )
        self.lr_scheduler = ParamScheduler(lr_schedule_config, total_steps=args.train_iters)

        # weight decay parameters
        self.wd_decr_style = args.wd_decr_style
        self.wd_begin = float(args.wd_begin)
        self.wd_end = float(args.wd_end)
        assert 0.0 <= self.wd_end <= self.wd_begin
        self.wd_decr_steps = args.train_iters
        wd_schedule_config = ScheduleConfig(
            begin=self.wd_begin,
            middle=self.wd_begin,
            end=self.wd_end,
            schedule_style=self.wd_decr_style,
            schedule_factor=1 if self.wd_end == self.wd_begin else float(self.wd_end/self.wd_begin),
        )
        self.wd_scheduler = ParamScheduler(wd_schedule_config, total_steps=args.train_iters)

        # grad accum step parameters
        self.ga_incr_style = args.grad_accum_step_incr_style
        self.ga_begin = int(args.ga_begin)
        self.ga_end = int(args.ga_end)
        assert 0.0 <= self.ga_begin <= self.ga_end
        self.ga_incr_steps = args.train_iters
        ga_schedule_config = ScheduleConfig(
            begin=self.ga_begin,
            middle=self.ga_begin,
            end=self.ga_end,
            schedule_style=self.ga_incr_style,
            schedule_factor=1 if self.ga_end == self.ga_begin else float(self.ga_end/self.ga_begin),
            data_type='int',
        )
        self.ga_scheduler = ParamScheduler(ga_schedule_config, total_steps=args.train_iters)

    def step(self, increment=1):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_lr = self.lr_scheduler.step(increment=increment)
        new_ga_step = self.ga_scheduler.step(increment=increment)
        new_wd = self.wd_scheduler.step(increment=increment)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
            if group["weight_decay"] != 0:
                group["weight_decay"] = new_wd
        return new_lr, new_ga_step, new_wd

    def state_dict(self):
        state_dict = {
            "total_steps": self.args.train_iters,
            "num_steps": self.num_steps,
            "lr_begin": self.lr_scheduler.config.begin,
            "lr_max": self.lr_scheduler.config.middle,
            "lr_warmup_steps": self.lr_scheduler.warmup_steps,
            "lr_decay_steps": self.lr_scheduler.schedule_steps,
            "lr_decay_factor": self.lr_scheduler.config.schedule_factor,
            "lr_decay_style": self.lr_scheduler.config.schedule_style,
            "wd_begin": self.wd_scheduler.config.begin,
            "wd_end": self.wd_scheduler.config.end,
            "wd_decr_steps": self.wd_scheduler.schedule_steps,
            "wd_decr_style": self.wd_scheduler.config.schedule_style,
            "ga_begin": self.ga_scheduler.config.begin,
            "ga_end": self.ga_scheduler.config.end,
            "ga_incr_steps": self.ga_scheduler.schedule_steps,
            "grad_accum_step_incr_style": self.ga_scheduler.config.schedule_style,
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
        # lr paras
        self.lr_decay_factor = self._check_and_set(self.lr_decay_factor, sd["lr_decay_factor"], "lr decay factor")
        self.lr_begin = self._check_and_set(self.lr_begin, sd["lr_begin"], "initial learning rate")
        self.lr_max = self._check_and_set(self.lr_max, sd["lr_max"], "maximum learning rate")
        self.lr_min = self.lr_max*self.lr_decay_factor
        self.lr_warmup_steps = self._check_and_set(self.lr_warmup_steps, sd["lr_warmup_steps"], "warmup iterations")
        self.lr_decay_steps = self._check_and_set(self.lr_decay_steps, sd["lr_decay_steps"], "number of lr_decay iterations")
        self.lr_decay_style = self._check_and_set(self.lr_decay_style, sd["lr_decay_style"], "learning rate decay style")
        lr_schedule_config = ScheduleConfig(
            begin=self.lr_begin,
            middle=self.lr_max,
            end=self.lr_min,
            schedule_style=self.lr_decay_style,
            schedule_factor=self.lr_decay_factor,
            warmup_ratio=self.lr_warmup_steps/self.args.train_iters,
            schedule_ratio=self.lr_decay_steps/self.args.train_iters,
        )
        self.lr_scheduler = ParamScheduler(lr_schedule_config, total_steps=self.args.train_iters)

        # wd paras
        if "wd_begin" in sd:
            self.wd_begin = self._check_and_set(self.wd_begin, sd["wd_begin"], "initial weight decay coefficient")
            self.wd_end = self._check_and_set(self.wd_end, sd["wd_end"], "final weight decay coefficient")
            self.wd_decr_steps = self._check_and_set(self.wd_decr_steps, sd["wd_decr_steps"], "total number of weight decay iterations",)
            self.wd_decr_style = self._check_and_set(self.wd_decr_style, sd["wd_decr_style"], "weight decay incr style")
            wd_schedule_config = ScheduleConfig(
                begin=self.wd_begin,
                middle=self.wd_begin,
                end=self.wd_end,
                schedule_style=self.wd_decr_style,
                schedule_factor=1 if self.wd_end == self.wd_begin else float(self.wd_end/self.wd_begin),
                schedule_ratio=self.wd_decr_steps/self.args.train_iters,
            )
            self.wd_scheduler = ParamScheduler(wd_schedule_config, total_steps=self.args.train_iters)

        # grad accum para
        if "ga_begin" in sd:
            self.ga_begin = self._check_and_set(self.ga_begin, sd["ga_begin"], "initial grad accum step")
            self.ga_end = self._check_and_set(self.ga_end, sd["ga_end"], "final grad accum step")
            self.ga_incr_steps = self._check_and_set(self.ga_incr_steps, sd["ga_incr_steps"], "total number of grad incr iterations",)
            self.ga_incr_style = self._check_and_set(self.ga_incr_style, sd["grad_accum_step_incr_style"], "grad accum step incr style")
            ga_schedule_config = ScheduleConfig(
                begin=self.ga_begin,
                middle=self.ga_begin,
                end=self.ga_end,
                schedule_style=self.ga_incr_style,
                schedule_factor=1 if self.ga_end == self.ga_begin else float(self.ga_end/self.ga_begin),
                schedule_ratio=self.ga_incr_steps/self.args.train_iters,
                data_type='int',
            )
            self.ga_scheduler = ParamScheduler(ga_schedule_config, total_steps=self.args.train_iters)

        # keep the training-process-ratio consistent in case of the total_steps changed
        # step_factor = float(self.args.train_iters/sd["total_steps"])
        step_factor = 1.0
        num_steps = int(sd["num_steps"] * step_factor)
        self.step(increment=num_steps)
        if self.rank == 0 and num_steps != sd["num_steps"]:
            print(f" > set num_steps value form {sd['num_steps']} to {num_steps}")

        return step_factor

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
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))

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
    plt.savefig("scheduler.png")
