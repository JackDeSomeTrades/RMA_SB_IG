from abc import ABC, abstractmethod
from isaacgym import gymutil
import gym
import torch
import warnings


class BaseTask(ABC, gym.Env):
    def __init__(self, cfg, sim_params, sim_device):

        self.cfg = cfg
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(
            sim_device)

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = sim_device
        else:
            self.device = 'cpu'
            warnings.warn("Warning : cpu is used")
        num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        if type(cfg.env.num_privileged_obs) == str:
            cfg.env.num_privileged_obs = eval(cfg.env.num_privileged_obs)
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers

        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.test_pos = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.env_done = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.reset_indices = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else:
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.infos = {}

    def close(self) :
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_observations(self):
        pass

    @abstractmethod
    def get_privileged_observations(self):
        pass
    
    