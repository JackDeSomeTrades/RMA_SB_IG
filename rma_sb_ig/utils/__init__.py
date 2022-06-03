import os
from rma_sb_ig.envs.base_task import BaseTask
import importlib


__MODEL_DICT__ = dict()


def env_gen(name:str):
    return __MODEL_DICT__[name]

def register(name:str):
    def register_function_fn(cls:type):
        if name in __MODEL_DICT__:
            raise ValueError(f"Name {name} already registered!")
        if not issubclass(cls, BaseTask):
            raise ValueError(f"Class {cls} is not a subclass of {BaseTask}")
        __MODEL_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file=='stable_baselines.py':
        module_name = file[:file.find('.py')]
        modules = importlib.import_module('rma_sb_ig.utils.'+module_name)

