from time import time
from warnings import WarningMessage
import numpy as np
import os

import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import gym.spaces as gymspace
import math

from rma_sb_ig.utils.helpers import *
from rma_sb_ig.envs.soto_forward_task import SotoForwardTask


class SotoRobotTask(SotoForwardTask):
    def __init__(self, *args,final_computation = False):
        self.final_computation = final_computation
        self.init_done = False
        super(SotoRobotTask, self).__init__(*args)
    def _init_observation_space(self):
        # Observation consists of : 43
        #     intrinsic - 22
        #                 dof_pos - 6 + 2 belt
        #                 dof_vel - 6 + 2 belt
        #                 distance_btw_conveyors - 1
        #                 distance sensors feedback(d1,d2) - 2
        #                 GC_x - 1
        #                 GC_y - 1
        #                 box_angle - 1
        #     extrinsic - 5
        #                 friction box- 1
        #                 friction belt - 1
        #                 Mass_box - 1
        #                 COM_x - 1
        #                 COM_y - 1
        #   previous_action
        #           - 8
        self.right_arm_index = self.dof_names.index('gripper_y_right')
        self.left_arm_index = self.dof_names.index('gripper_y_left')
        dist_max = self.upper_bounds_joints[self.right_arm_index] - self.lower_bounds_joints[self.right_arm_index]
        limits_low = np.array(
            list(self.lower_bounds_joints) +
            list(-self.joint_velocity)+    # minimum values of joint velocities
            [0] +
            [0]*2+
            [-1.5] * 2 +
            [-np.pi]+
            list(self.lower_bounds_joints) +
            [self.cfg.domain_rand.friction_static_range[0]]*2 +
            [self.cfg.domain_rand.mass_box[0]] +
            [-1.5]*2
        )

        limits_high = np.array(
            list(self.upper_bounds_joints) +
            list(self.joint_velocity)+    # maximum values of joint velocities# distance max btw 2 conveyors +
            [dist_max] +
            [1.5]*2 +
            [1.5] * 2 +
            [np.pi] +
            list(self.upper_bounds_joints) +
            [self.cfg.domain_rand.friction_static_range[1]]*2 +
            [self.cfg.domain_rand.mass_box[1]] +
            [1.5]*2
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

        self.lower_bounds_joint_tensor = torch.tensor(self.lower_bounds_joints, dtype=torch.float, device=self.device).expand(self.num_envs, self.num_dofs)
        self.upper_bounds_joint_tensor = torch.tensor(self.upper_bounds_joints, dtype=torch.float, device=self.device).expand(self.num_envs, self.num_dofs)
        self.torque_force_bound = torch.tensor(self.motor_strength, dtype=torch.float, device=self.device).expand(self.num_envs,self.num_dofs)
        self.joint_velocity_bound = torch.tensor(self.joint_velocity, dtype=torch.float, device=self.device).expand(self.num_envs,self.num_dofs)


        lb = self.lower_bounds_joints
        ub = self.upper_bounds_joints
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
        #     intrinsic - 22
        #                 dof_pos - 6 + 2 belt
        #                 dof_vel - 6 + 2 belt
        #                 distance_btw_conveyors - 1
        #                 distance sensors feedback(d1,d2) - 2
        #                 GC_x - 1
        #                 GC_y - 1
        #                 box_angle - 1
        #     extrinsic - 5
        #                 friction box- 1
        #                 friction belt - 1
        #                 Mass_box - 1
        #                 COM_x - 1
        #                 COM_y - 1
        #   previous_action - 8
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:8] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[8:16] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[16:17] = 0.05
        noise_vec[17:19] = 0.02
        noise_vec[19:21] = noise_scales.distance_measurements * \
            noise_level
        noise_vec[21:22] = 0.05
        noise_vec[22:30] = noise_scales.action * \
            noise_level * self.obs_scales.action
        noise_vec[30:32] = 0.05 #friction
        noise_vec[32:33] = 0.3
        noise_vec[33:35] = 0.05  # Mass and COM and GC
        return noise_vec

    def compute_observations(self):
        #     intrinsic - 22
        #                 dof_pos - 6 + 2 belt
        #                 dof_vel - 6 + 2 belt
        #                 distance_btw_conveyors - 1
        #                 distance sensors feedback(d1,d2) - 2
        #                 GC_x - 1
        #                 GC_y - 1
        #                 box_angle - 1
        #     extrinsic - 5
        #                 friction box- 1
        #                 friction belt - 1
        #                 Mass_box - 1
        #                 COM_x - 1
        #                 COM_y - 1
        #   previous_action - 8
        self.get_depth_sensors()
        self.distance_sensors = torch.clip(self.distance_sensors, 0.025, 1.5)
        distance_btw_arms = torch.abs(self.dof_pos[:, self.right_arm_index] - (0.7 - self.dof_pos[:, self.left_arm_index]))
        self.X_t = torch.cat((self.dof_pos,
                              self.dof_vel,
                              distance_btw_arms.unsqueeze(-1),
                              self.distance_sensors,
                              self.box_pos[..., :2],
                              self.box_angle.unsqueeze(-1)), dim=-1)

        E_t = torch.cat((
            self.soto_fric,
            self.box_fric,
            self.box_masses.unsqueeze(-1),
            self.box_com_x.unsqueeze(-1),
            self.box_com_y.unsqueeze(-1)
        ), dim=-1)

        self.obs_buf = torch.cat((self.X_t,
                                  self.last_actions,
                                  E_t), dim=-1)
        # add noise if needed
        noise = self._get_noise_scale_vec(self.cfg)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -1) * noise
    # -------------- Reward functions begin below: --------------------------------#

    def _reward_turning_velocity(self):
        reward = torch.square(self.rigid_body_tensor[:,self.gripper_x_id,12])
        return reward

    def _reward_turn(self):
        value = torch.remainder(self.commands.squeeze(-1)-self.box_angle,torch.pi)
        value = torch.where(value <= torch.pi/2,value,torch.pi-value)
        self.angle_error = value
        angle_error = torch.square(value)
        self.env_done = value < 0.05
        reward = torch.exp(-angle_error / self.cfg.rewards.tracking_angle)
        return reward

    def _reward_ecartement(self):
        d1 = self.box_dim[:,0]*torch.cos(self.angle_error) + self.box_dim[:,1]*torch.sin(self.angle_error)-0.25
        d2 = torch.abs(self.dof_pos[:, self.right_arm_index] - (0.7 - self.dof_pos[:, self.left_arm_index]))
        reward = torch.exp(-torch.square(d1-d2)/ self.cfg.rewards.tracking_arms)
        return reward

    def _reward_termination(self):
        # Terminal reward / penalty
        reward = torch.where(self.env_done,1,0)
        return reward

    def _reward_velocity(self):
        value = torch.square(self.dof_vel[:,self.right_conv_belt_id]+self.dof_vel[:,self.left_conv_belt_id])
        reward = torch.abs(torch.max(self.dof_vel[:,self.right_conv_belt_id],self.dof_vel[:,self.left_conv_belt_id]))*torch.exp(-value/self.cfg.rewards.velocity)
        return reward

    def _reward_distance(self):
        angle = torch.remainder(self.commands.squeeze(-1) - self.box_angle, torch.pi)-torch.pi/2
        dist = torch.abs(self.box_dim[:,1]*torch.sin(angle))+torch.abs(self.box_dim[0]*torch.cos(angle))
        dist2 = torch.abs(
            self.dof_pos[:, self.right_arm_index] - (0.7 - self.dof_pos[:, self.left_arm_index]))
        reward = torch.exp(torch.square(dist-dist2))
        return reward

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)


    # def _reward_geometric_center(self):
    #     reward = torch.norm(self.box_pos - self.box_init_pos,dim=1)
    #     return reward
    # def _reward_work(self):
    #     diff_joint_pos = self.actions - self.last_actions
    #     # torque_transpose = torch.transpose(self.torques, 0, 1)
    #     # this is the L1 norm
    #     reward = torch.abs(torch.sum(torch.inner(
    #         self.torques_forces, diff_joint_pos), dim=1))
    #     return reward
    def _reward_z_position(self):
        reward = torch.abs(self.box_pos[:,2] - self.box_init_pos[:,2])
        return reward

    # ---------------Reward functions end here ---------------------- #

    def _get_random_boxes(self, l_limit, w_limit, h_limit):
        length = random.uniform(l_limit[0], l_limit[1])
        width = random.uniform(w_limit[0], w_limit[1])
        height = random.uniform(h_limit[0], h_limit[1])
        return (length, width, height)

    def get_depth_sensors(self):
        q = self.box_quat.resize(self.num_envs,1,4).expand(-1,3,-1).resize(3*self.num_envs,4)
        v = self.box_init_axis.resize(3*self.num_envs,3)
        box_axis = quat_rotate(q, v)
        #self.box_dim

        C = self.box_root_state[:,:3].resize(self.num_envs,1,3).expand(-1,3,-1).resize(3*self.num_envs,3)

        P1 = self.rigid_body_tensor[:, self.conveyor_left_id, :3].resize(self.num_envs,1,3).expand(-1,3,-1).resize(3*self.num_envs,3)
        P1[:,2] += 0.1
        P2 = self.rigid_body_tensor[:, self.conveyor_right_id, :3].resize(self.num_envs,1,3).expand(-1,3,-1).resize(3*self.num_envs,3)
        P2[:,2] += 0.1
        d = quat_rotate(self.rigid_body_tensor[:, self.conveyor_right_id, 3:7],self.gripper_init_x_axis).resize(self.num_envs,1,3).expand(-1,3,-1).resize(3*self.num_envs,3)
        #parallel =  torch.abs(torch.sum(d*box_axis,dim = 1))
        r1 = torch.sum(box_axis*(C-P1),dim = 1)
        r2 = torch.sum(box_axis*(C-P2), dim = 1)
        s = torch.sum(box_axis*d, dim = 1)

        t0_1 = (r1 + torch.flatten(self.box_dim)/2)/s
        t1_1 = (r1 - torch.flatten(self.box_dim)/2)/s

        t0_2 = (r2 + torch.flatten(self.box_dim)/2)/s
        t1_2 = (r2 - torch.flatten(self.box_dim)/2)/s
        t1near,_ = torch.max(torch.min(t0_1,t1_1).resize(self.num_envs,3),dim=1)
        t2near,_ = torch.max(torch.min(t0_2, t1_2).resize(self.num_envs,3),dim=1)

        self.distance_sensors[:,0] = t1near
        self.distance_sensors[:,1] = t2near