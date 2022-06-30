from time import time
from warnings import WarningMessage
import numpy as np
import os
import re

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
from rma_sb_ig.envs.forward_task import VarForwardTask


class V0VarLeggedRobotTask(VarForwardTask):
    def __init__(self, *args):
        super(V0VarLeggedRobotTask, self).__init__(*args)

    def _init_observation_space(self):
        """
        Observation function for a variable v0 robot (generalised to x number of legs; x=4, 5 or 6) consists of the following parameters:
                dof_pos           -    num_legs * 3
                dof_ vel          -    num_legs * 3
                roll              -    1
                yaw               -    1
                foot_contact      -    num_legs
                heading_deviation -    1
                previous_action   -    num_legs * 3
                --------
                Mass              -    1
                com_x             -    1
                com_y             -    1
                com_z             -    1
                friction          -    1
                terrain_height    -    num_legs
                projected_gravity -    3
                structure_dist    -    num_legs  ( TODO: for more than num_legs. excess legs to be represented by -1)
                structure_angle   -    num_legs  ( TODO: for more than num_legs. excess legs represented by 0)

                Total state space size ---- 43 + 20 = 63

        :return: obs_space
        """
        # TODO: Generalise for more than 4 legs. For the moment, running with 4 legs.

        limits_low = np.array(
            self.lower_bounds_joints +   # dof_pos
            [0] * (self.cfg.asset.num_legs * 3) +    # minimum values of joint velocities (dof_vels)
            [-math.inf] +   # roll
            [-math.inf] +   # yaw
            [0] * self.cfg.asset.num_legs +  # foot_contact
            [-math.inf] +  # heading deviation
            self.lower_bounds_joints +   # previous action
            # ------------#
            [0] +             # Mass
            [-math.inf] * 3 +       # com_ x, y, z
            [self.cfg.domain_rand.friction_range[0]] +   # friction
            [0] * self.cfg.asset.num_legs +   # local terrain height
            [-math.inf] * 3 +   # Projected Gravity
            [-1] * self.cfg.asset.num_legs +      # TODO: Make this generic for over 4 legs. At the moment, it's only for 4
            [-math.inf] * self.cfg.asset.num_legs    # structure angle
        )
        limits_high = np.array(
            self.upper_bounds_joints +
            self.upper_bound_joint_velocities +
            [math.inf] +
            [math.inf] +
            [1] * self.cfg.asset.num_legs +
            [math.inf] +
            self.upper_bounds_joints +
            #  ------------------ #
            [math.inf] +      # Mass
            [math.inf] * 3 +  # com_ x, y, z
            [self.cfg.domain_rand.friction_range[1]] +
            [math.inf] * self.cfg.asset.num_legs +  # local terrain height
            [0] * 3 +         # Projected Gravity
            [math.inf] * self.cfg.asset.num_legs +
            [math.inf] * self.cfg.asset.num_legs
        )
        obs_space = gymspace.Box(limits_low, limits_high, dtype=np.float32)
        return obs_space

    def _init_action_space(self):
        """
        Upper and lower bounds of all 12 joints  of the robot extracted automatically from the URDF.
        Extraction function and values are defined in the accompanying scene class.
        Values can be found under <joint ...> <limit ... upper="" lower=""> </joint>

        :return: act_space -> gym.space.Box
        """
        lb = np.array(self.lower_bounds_joints)
        ub = np.array(self.upper_bounds_joints)
        act_space = gymspace.Box(lb, ub, dtype=np.float32)

        return act_space

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12:24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[24:26] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[26:30] = 0.
        noise_vec[30] = 0.
        noise_vec[31:43] = 0.  # Previous action.
        noise_vec[43:48] = 0.  # Mass, COM and friction
        noise_vec[48:52] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return noise_vec

    def compute_observations(self):
        """Overrides the base class observation computation to bring it in line with observations proposed in the RMA
        paper. The observation consists of two major parts - the environment variables and the state-action pair."""
        self.compute_heading_deviation()
        feet_contact_switches = self._get_foot_status()
        local_terrain_height = self._get_local_terrain_height()
        structure_dist, structure_angle = self._get_asset_struct_details()

        self.X_t = torch.cat((self.dof_pos,
                              self.dof_vel,
                              self.base_rpy[0].unsqueeze(1),
                              self.base_rpy[1].unsqueeze(1),
                              feet_contact_switches,
                              self.heading_deviation
                              ), dim=-1)
        E_t = torch.cat((
            self.body_masses.unsqueeze(-1),
            self.full_body_com_x.unsqueeze(-1),
            self.full_body_com_y.unsqueeze(-1),
            self.full_body_com_z.unsqueeze(-1),
            self.friction_coeffs.squeeze(-1),
            local_terrain_height,
            self.projected_gravity,    # TODO: Check this to see if the dimensions are right.
            structure_dist,
            structure_angle
        ), dim=-1)

        self.obs_buf = torch.cat((self.X_t,
                                  self.last_actions,
                                  E_t), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def compute_heading_deviation(self):
        commands_xy = self.commands[:, :2]
        command_normalized = torch.nan_to_num(
            (commands_xy.T / torch.norm(commands_xy, dim=1, p=2)).T
        )
        base_xy_velocity = self.base_lin_vel[:, :3]
        forward_orientation = torch.zeros_like(base_xy_velocity)
        forward_orientation[:, 0] = 1.0
        forward_orientation_quat = quat_rotate(self.base_quat, forward_orientation)
        forward_orientation_quat = normalize(forward_orientation_quat)[:, :2]

        self.heading_deviation = torch.acos((command_normalized * forward_orientation_quat).sum(dim=1))
        self.heading_deviation = (torch.sign(forward_orientation_quat[:, 1]) * self.heading_deviation).reshape((-1, 1))

        return self.heading_deviation

    def _get_asset_struct_details(self):
        self.asset_base_joint_positions = [[[float(k) for k in re.split('[\s]', x)] for x in lst_] for lst_ in self.asset_base_joint_positions]

        dist = lambda a: a[0]**2 + a[1]**2
        angle = lambda a: math.atan2(a[1], a[0])

        dist_leg = [[dist(a) for a in lst_] for lst_ in self.asset_base_joint_positions]
        ang_leg = [[angle(a) for a in lst_] for lst_ in self.asset_base_joint_positions]

        dist_leg = torch.cuda.FloatTensor(dist_leg)
        ang_leg = torch.cuda.FloatTensor(ang_leg)

        return dist_leg, ang_leg


    def _get_foot_status(self):
        # foot_status = torch.zeros_like(self.feet_indices)
        # feet_forces = self.contact_forces[:, self.feet_indices, :2]
        # for foot_index in self.feet_indices:
        #     contact = self.contact_forces[:, foot_index, 2]
        #     if contact > 0.:
        #         foot_status[foot_index] = 1.

        foot_status = torch.ones(self.cfg.env.num_envs, 4, device=self.device)
        feet_forces = self.contact_forces[:, self.feet_indices, 2]
        feet_switches = torch.where(feet_forces == 0., torch.tensor(0.0, device=self.device), foot_status)

        return feet_switches

    def _get_local_terrain_height(self):
        # Getting local terrain height is different for this robot compared to the A1. In this, we're extracting the
        # COM-z data for each of the robots' feet

        # get rigid body properties for each handle.
        # Extract props of the feet using feet indices.
        # Extract COMM-x from feet props. That is the local terrain height.
        feet_pos_all = []
        for i in range(self.num_envs):
            feet_pos_per_env = []
            env_handle = self.gym.get_env(self.sim, i)
            actor_handle = self.gym.find_actor_handle(env_handle, self.cfg.asset.name)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            for foot_index in self.feet_indices:
                feet_pos_per_env.append(body_props[foot_index].com.z)
            feet_pos_all.append(feet_pos_per_env)

        local_terrain_height = torch.cuda.FloatTensor(feet_pos_all)

        return local_terrain_height

        # -------------- Reward functions begin below: --------------------------------#

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_base_height(self):
        # Make sure the robot body maintains a minimum distance from the ground based on the z center of mass.
        base_height = torch.mean(self.body_com_z.unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        reward = torch.square(self.base_lin_vel[:, 2])
        return reward

