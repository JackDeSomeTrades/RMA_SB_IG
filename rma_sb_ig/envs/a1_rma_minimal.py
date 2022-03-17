from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import box
import gym.spaces as gymspace
import math
import torch

# import torch
# from torch import Tensor
# from typing import Tuple, Dict
from rma_sb_ig.utils.helpers import *
from rma_sb_ig.envs.base_task import BaseTask
from rma_sb_ig.utils.scene import EnvScene


class A1LeggedRobotTaskMinimal():
    def __init__(self):

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                                           device=self.device, requires_grad=False, )
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(),
                             "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(),
                             "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(),
                             "action_rate": torch_zeros(), "hip": torch_zeros()}

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True

    def create_sim(self):
        self.up_axis_idx = set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)



