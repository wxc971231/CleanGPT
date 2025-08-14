import os
from abc import ABC, abstractmethod
from train.config import parse_args

class BaseExperimentConfig(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = ""

    def get_base_args(self):
        base_args = parse_args()
        base_args.world_size = int(os.environ.get("WORLD_SIZE", default='1'))
        return base_args

    @abstractmethod
    def get_args_ready(self):
        pass