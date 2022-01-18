from time import time
from warnings import WarningMessage
import numpy as np
import os
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from box import Box

# import torch
# from torch import Tensor
# from typing import Tuple, Dict
from utils.helpers import *
from envs.base_task import BaseTask
from utils.scene import EnvScene


class A1LeggedRobotTask(BaseTask, EnvScene):
    def __init__(self, cfg):
        config = cfg['task_config']
        args = get_args()  # needs to be done only to follow gymutils implements it this way. Future work: to redo this.
        sim_params = parse_sim_params(args, config)
        config = Box(config)
        BaseTask.__init__(self, cfg=config, sim_params=sim_params, sim_device=args.sim_device)
        EnvScene.__init__(self, cfg=config, physics_engine=args.physics_engine, sim_device=args.sim_device, headless=args.headless, sim_params=sim_params)
        pass

    def close(self):
        pass

    def reset(self, env_ids):
        pass

    def env_is_wrapped(self):
        pass

    def get_observations(self):
        pass

    def get_privileged_observations(self):
        pass