from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym import gymapi
from isaacgym.torch_utils import *
import gym.spaces as gymspace
import math

from rma_sb_ig.utils.helpers import *
from rma_sb_ig.envs.soto_forward_task import SotoForwardTask


class SotoRobotTask(SotoForwardTask):
    def __init__(self, *args):

        super(SotoRobotTask, self).__init__(*args)

    def _init_observation_space(self):
        """
        Observation consists of :
            intrinsic
                        dof_pos - 7
                        dof_vel - 7
                        previous_action - 7
                        distance_btw_conveyors - 1
            extrinsic
                        friction box/belt - 2
                        dynamic friction - 1
                        Mass_box - 1
                        COM_x - 1
                        COM_y - 1
                        width-length-height_box - 3
                        distance sensors feedback(d1,d2) - 2
                        box_angle - 1
        :return: obs_space
        """

        # TODO : Do not forget to add motors control when they will be available
        index = self.soto_indexs[0]
        dist_max = self.upper_bounds_joints[index] - self.lower_bounds_joints[index]
        limits_low = np.array(
            list(self.lower_bounds_joints) +
            [0] * self.num_dofs +    # minimum values of joint velocities
            list(self.lower_bounds_joints) +
            [0] +


            [self.cfg.domain_rand.friction_range[0]]*2 +
            [self.cfg.domain_rand.friction_range[0]] +
            [self.cfg.domain_rand.mass_box[0]] +
            [0] +
            [0] +
            [self.cfg.domain_rand.width_box[0]] +
            [self.cfg.domain_rand.length_box[0]] +
            [self.cfg.domain_rand.height_box[0]] +

            [0]*2 +
            [0]
        )

        limits_high = np.array(
            list(self.lower_bounds_joints) +
            [0] * self.num_dofs +    # minimum values of joint velocities
            list(self.lower_bounds_joints) +
            # distance max btw 2 conveyors +
            [dist_max] +


            [self.cfg.domain_rand.friction_range[1]]*2 +
            [self.cfg.domain_rand.friction_range[1]] +
            [self.cfg.domain_rand.mass_box[1]] +

            [self.cfg.domain_rand.length_box[1]-self.cfg.domain_rand.length_box[0]] +
            [self.cfg.domain_rand.width_box[1]-self.cfg.domain_rand.width_box[0]] +

            [self.cfg.domain_rand.width_box[1]] +
            [self.cfg.domain_rand.length_box[1]] +
            [self.cfg.domain_rand.height_box[1]] +

            [2.0]*2 +  # supposed length of grippers
            [2*np.pi]
        )

        obs_space = gymspace.Box(
            limits_low, limits_high, dtype=np.float32)
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

        noise_vec[:12] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[12:24] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[24:26] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[26:30] = 0.
        noise_vec[30:42] = 0.  # Previous action.
        noise_vec[42:45] = 0.  # Mass and COM
        noise_vec[45:57] = noise_scales.motor_strength * noise_level
        noise_vec[58] = noise_scales.height_measurements * \
            noise_level * self.obs_scales.height_measurements

        return noise_vec

    def compute_observations(self):
        """Overrides the base class observation computation to bring it in line with observations proposed in the RMA
        paper. The observation consists of two major parts - the environment variables and the state-action pair."""
        # self.compute_heading_deviation()
        feet_contact_switches = self._get_foot_status()
        local_terrain_height = self._get_local_terrain_height()

        self.X_t = torch.cat((self.dof_pos,
                              self.dof_vel,
                              self.base_rpy[0].unsqueeze(1),
                              self.base_rpy[1].unsqueeze(1),
                              feet_contact_switches
                              ), dim=-1)
        E_t = torch.cat((
            self.body_masses.unsqueeze(-1),
            self.body_com_x.unsqueeze(-1),
            self.body_com_y.unsqueeze(-1),
            self.torque_limits,
            self.friction_coeffs.squeeze(-1),
            local_terrain_height.unsqueeze(-1)
        ), dim=-1)

        self.obs_buf = torch.cat((self.X_t,
                                  self.last_actions,
                                  E_t), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                             1) * self.noise_scale_vec

    def compute_heading_deviation(self):
        commands_xy = self.commands[:, :2]
        command_normalized = torch.nan_to_num(
            (commands_xy.T / torch.norm(commands_xy, dim=1, p=2)).T
        )
        base_xy_velocity = self.base_lin_vel[:, :3]
        forward_orientation = torch.zeros_like(base_xy_velocity)
        forward_orientation[:, 0] = 1.0
        forward_orientation_quat = quat_rotate(
            self.base_quat, forward_orientation)
        forward_orientation_quat = normalize(forward_orientation_quat)[:, :2]

        self.heading_deviation = torch.acos(
            (command_normalized * forward_orientation_quat).sum(dim=1))
        self.heading_deviation = (torch.sign(
            forward_orientation_quat[:, 1]) * self.heading_deviation).reshape((-1, 1))

        return self.heading_deviation

    # -------------- Reward functions begin below: --------------------------------#

    def _reward_forward(self):
        base_x_velocity = self.base_lin_vel[:, 0]
        MAX_FWD_VEL = 1.0
        forward = quat_apply(self.base_quat, self.forward_vec)
        # max_fwd_vel_tensor = torch.full_like(base_x_velocity, MAX_FWD_VEL)
        # reward = torch.abs(base_x_velocity - max_fwd_vel_tensor)
        diff = base_x_velocity - forward[:, 0]
        reward = torch.linalg.norm(
            diff.unsqueeze(-1), dim=1, ord=1)   # L1 norm

        return reward

    def _reward_maintain_forward(self):
        # Rewards forward motion at limited values (v_x)
        base_x_velocity = self.base_lin_vel[:, 0]
        MAX_FWD_VEL = 1.0
        max_fwd_vel_tensor = torch.full_like(base_x_velocity, MAX_FWD_VEL)
        reward = torch.fmin(base_x_velocity, max_fwd_vel_tensor)
        # print(" forward x vel", base_x_velocity)
        # print(" forward reward", reward)
        reward[reward < 0.0] = 0.0   # TODO: Check if this is right.
        # print(f"forward reward * {self.cfg.rewards.scales.forward}", reward)
        return reward

    def _reward_lateral_movement_rotation(self):
        # penalises lateral motion (v_y) and limiting angular velocity yaw
        reward = self.base_lin_vel[:, 1].pow(2).unsqueeze(-1).sum(
            dim=1) + self.base_ang_vel[:, 2].pow(2).unsqueeze(-1).sum(dim=1)  # TODO check with 1
        # print(f"lateral reward * {self.cfg.rewards.scales.lateral_movement_rotation}", reward)
        return reward

    def _reward_orientation(self):
        # orientation = self.projected_gravity[:, :2]
        orientation = torch.stack([self.base_rpy[0], self.base_rpy[1]], dim=1)
        reward = torch.sum(torch.square(orientation), dim=1)
        return reward

    def _reward_work(self):
        diff_joint_pos = self.actions - self.last_actions
        # torque_transpose = torch.transpose(self.torques, 0, 1)
        # this is the L1 norm
        reward = torch.abs(torch.sum(torch.inner(
            self.torques, diff_joint_pos), dim=1))
        return reward

    def _reward_ground_impact(self):
        reward = torch.sum(torch.square(self.contact_forces[:, self.feet_indices, 2] -
                                        self.last_contact_forces[:, self.feet_indices, 2]), dim=1)  # only taking into account the vertical reaction, might need to check if parallel ground reaction makes sense.
        return reward

    def _reward_smoothness(self):
        reward = torch.sum(torch.square(self.torques-self.last_torques), dim=1)
        return reward

    def _reward_action_magnitude(self):
        reward = torch.sum(torch.square(self.actions), dim=1)
        return reward

    def _reward_joint_speed(self):
        reward = torch.sum(torch.square(self.last_dof_vel), dim=1)
        return reward

    def _reward_z_acceleration(self):
        reward = torch.square(self.base_lin_vel[:, 2])
        return reward

    def _reward_foot_slip(self):
        reward = 0  # TODO: Fix this
        return reward

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        reward = torch.square(self.base_lin_vel[:, 2])
        return reward

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_height(self):
        # Make sure the robot body maintains a minimum distance from the ground based on the z center of mass.
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(
            1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = - \
            (self.dof_pos -
             self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos -
                          self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits *
             self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact,
                                dim=1)  # reward only on first contact with the ground
        # no reward for zero command
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # ---------------Reward functions end here ---------------------- #

    def _get_foot_status(self):
        # foot_status = torch.zeros_like(self.feet_indices)
        # feet_forces = self.contact_forces[:, self.feet_indices, :2]
        # for foot_index in self.feet_indices:
        #     contact = self.contact_forces[:, foot_index, 2]
        #     if contact > 0.:
        #         foot_status[foot_index] = 1.

        foot_status = torch.ones(self.cfg.env.num_envs, 4, device=self.device)
        feet_forces = self.contact_forces[:, self.feet_indices, 2]
        feet_switches = torch.where(feet_forces == 0., torch.tensor(
            0.0, device=self.device), foot_status)

        return feet_switches

    def _get_local_terrain_height(self):
        local_terrain_height = self.measured_heights
        # this gives the measured heights of all points below the body of the robot. Max of local terrain height is just
        # the max of each of the heights of the envs. This is in contrast to the paper which defines terrain height as
        # the max of the height below the robot feet.
        local_terrain_height = torch.max(local_terrain_height, dim=-1)[0]

        return local_terrain_height

    def _get_random_boxes(self, l_limit, w_limit, h_limit):
        length = random.uniform(l_limit[0], l_limit[1])
        width = random.uniform(w_limit[0], w_limit[1])
        height = random.uniform(h_limit[0], h_limit[1])
        return (length, width, height)
