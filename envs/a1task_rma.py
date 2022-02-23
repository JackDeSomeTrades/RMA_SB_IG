from time import time
from warnings import WarningMessage
import numpy as np
import os

import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import box
import gym.spaces as gymspace
import math

# import torch
# from torch import Tensor
# from typing import Tuple, Dict
from utils.helpers import *
from envs.base_task import BaseTask
from utils.scene import EnvScene


class A1LeggedRobotTask(EnvScene, BaseTask):
    def __init__(self, *args):
        config = args[0][0]
        sim_params = args[0][1]
        args = args[0][2]

        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        BaseTask.__init__(self, cfg=config, sim_params=sim_params, sim_device=args.sim_device)
        EnvScene.__init__(self, cfg=config, physics_engine=args.physics_engine, sim_device=args.sim_device, headless=args.headless, sim_params=sim_params)
        set_seed(config.seed)

        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = self.cfg.rewards.scales.to_dict()
        self.command_ranges = self.cfg.commands.ranges.to_dict()
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self.observation_space = self._init_observation_space()
        self.action_space = self._init_action_space()

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
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.infos = [{} for _ in range(self.num_envs)]
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)  # new addition for RMA
        self.last_contact_forces = torch.zeros_like(self.contact_forces)  # new addition for RMA
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        # self.base_lin_vel = self.root_states[:, 7:10]
        # self.base_ang_vel = self.root_states[:, 10:13]
        self.base_rpy = get_euler_xyz(self.base_quat)   # provides (r, p, y) tuple of the base torso with each r,p,y of size num_envs

        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.base_lin_vel = quat_rotate(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate(self.base_quat, self.root_states[:, 10:13])

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # initialise reward functions here
        self._prepare_reward_function()
        self.init_done = True

    def _init_observation_space(self):
        """
        Observation consists of :
                        dof_pos - 12
                        dof_vel - 12
                        roll - 1
                        pitch - 1
                        feet_contact_switches - 4
                        previous_actions -12
                        Mass - 1
                        COM_x - 1
                        COM_y - 1
                        Motor Strength - 12
                        Friction - 1
                        Local terrain height - 1
        :return: obs_space
        """
        limits_low = np.array(
            self.lower_bounds_joints +
            [0] * 12 +    # minimum values of joint velocities
            [-math.inf] +
            [-math.inf] +
            [0] * 4 +
            self.lower_bounds_joints +
            [0] +
            [-math.inf] +
            [-math.inf] +
            [motor_strength * self.cfg.domain_rand.motor_strength_range[0] for motor_strength in self.motor_strength] +
            [self.cfg.domain_rand.friction_range[0]] +
            [0]
        )
        limits_high = np.array(
            self.upper_bounds_joints +
            self.upper_bound_joint_velocities +
            [math.inf] +
            [math.inf] +
            [1] * 4 +
            self.upper_bounds_joints +
            [math.inf] +
            [math.inf] +
            [math.inf] +
            [motor_strength * self.cfg.domain_rand.motor_strength_range[1] for motor_strength in self.motor_strength] +
            [self.cfg.domain_rand.friction_range[1]] +
            [math.inf]
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

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

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
        noise_vec[30:42] = 0.  # Previous action.
        noise_vec[42:45] = 0.  # Mass and COM
        noise_vec[45:57] = noise_scales.motor_strength * noise_level
        noise_vec[58] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return noise_vec

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

                Args:
                    actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
                """
        # but SB3 requires the step return to be in the form : obs, rews, dones, info
        # infos is info, dones is reset_buf, rews is rew_buf,
        # Edit 2:More SB3 woes. SB3 outputs an ndarray after processing, but this requires a torch tensor. Need to
        # perform sanity checks before processing action.
        flag = 0
        print("actions", actions)

        if type(actions) == np.ndarray:
            # For SB3 compatibility
            flag = 1
            actions = torch.tensor(actions, device=self.device)

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            return self.obs_buf, self.rew_buf, self.reset_buf, self.infos, self.privileged_obs_buf
        if flag:
            # For SB3 compatibility
            rew_buf = self.rew_buf.detach().cpu().numpy()
            obs_buf = self.obs_buf.detach().cpu().numpy()
            dones = self.reset_buf.detach().cpu().numpy()
        else:
            rew_buf = self.rew_buf
            obs_buf = self.obs_buf
            dones = self.reset_buf

        print("#", "--"*50, "#")

        return obs_buf, rew_buf, dones, self.infos

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type == "V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        # self.base_lin_vel[:] = self.root_states[:, 7:10]
        # self.base_ang_vel[:] = self.root_states[:, 10:13]
        self.base_lin_vel[:] = quat_rotate(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate(self.base_quat, self.root_states[:, 10:13])  # TODO: Check this
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_rpy = get_euler_xyz(self.base_quat)

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            # self._push_robots()
            """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
                    """
            max_vel = self.cfg.domain_rand.max_push_vel_xy
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        print("base_lin_vel", self.base_lin_vel)
        print("base_ang_vel", self.base_ang_vel)

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_torques[:] = self.torques[:]  # for RMA
        self.last_contact_forces[:] = self.contact_forces[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # if env_ids.shape[0] != 0:
        #     print("env_ids:", env_ids)
        #     print("commands:", self.commands[env_ids, :])
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def compute_observations(self):
        """Overrides the base class observation computation to bring it in line with observations proposed in the RMA
        paper. The observation consists of two major parts - the environment variables and the state-action pair."""
        # self.compute_heading_deviation()
        feet_contact_switches = self._get_foot_status()
        local_terrain_height = self._get_local_terrain_height()

        X_t = torch.cat((self.dof_pos,
                         self.dof_vel,
                         self.base_rpy[0].unsqueeze(1),
                         self.base_rpy[1].unsqueeze(1),
                         # self.projected_gravity[:, 0].unsqueeze(1),
                         # self.projected_gravity[:, 1].unsqueeze(1),
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

        self.obs_buf = torch.cat((X_t,
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

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    # -------------- Reward functions begin below: --------------------------------#

    def _reward_forward(self):
        # Rewards forward motion at limited values (v_x)
        base_x_velocity = self.base_lin_vel[:, 0]
        MAX_FWD_VEL = 1.0
        max_fwd_vel_tensor = torch.full_like(base_x_velocity, MAX_FWD_VEL)
        reward = torch.fmin(base_x_velocity, max_fwd_vel_tensor)
        # print(" forward x vel", base_x_velocity)
        # print(" forward reward", reward)
        reward[reward < 0.0] = 0.0   # TODO: Check if this is right.
        print(f"forward reward * {self.cfg.rewards.scales.forward}", reward)
        return reward

    def _reward_lateral_movement_rotation(self):
        # penalises lateral motion (v_y) and limiting angular velocity yaw
        reward = self.base_lin_vel[:, 1].pow(2).unsqueeze(-1).sum(dim=1) + self.base_ang_vel[:, 2].pow(2).unsqueeze(-1).sum(dim=1)   #TODO check with 1
        print(f"lateral reward * {self.cfg.rewards.scales.lateral_movement_rotation}", reward)
        return reward

    def _reward_orientation(self):
        # orientation = self.projected_gravity[:, :2]
        orientation = torch.stack(self.base_rpy[0], self.base_rpy[1], dim=1)
        reward = torch.sum(torch.square(orientation), dim=1)
        return reward

    def _reward_work(self):
        diff_joint_pos = self.actions - self.last_actions
        # torque_transpose = torch.transpose(self.torques, 0, 1)
        reward = torch.abs(torch.sum(torch.inner(self.torques, diff_joint_pos), dim=1))     # this is the L1 norm
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
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
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
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
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
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
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
        feet_switches = torch.where(feet_forces == 0., torch.tensor(0.0, device=self.device), foot_status)

        return feet_switches

    def _get_local_terrain_height(self):
        local_terrain_height = self.measured_heights
        # this gives the measured heights of all points below the body of the robot. Max of local terrain height is just
        # the max of each of the heights of the envs. This is in contrast to the paper which defines terrain height as
        # the max of the height below the robot feet.
        local_terrain_height = torch.max(local_terrain_height, dim=-1)[0]

        return local_terrain_height

    def check_termination(self):
        """ Check if environments need to be reset. Sets up the dones for the return values of step.
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
                    Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
                    [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
                    Logs episode info
                    Resets some buffers

                Args:
                    env_ids (list[int]): List of environment ids which must be reset
                """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # fill infos
        # convert tensor to list -
        env_ids_list = env_ids.tolist()
        for env_id in env_ids_list:
            self.infos[env_id]["episode"] = {}
            for key in self.episode_sums.keys():
                self.infos[env_id]["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_id]) / self.max_episode_length_s
                self.infos[env_id]["episode"]["r"] = self.rew_buf.clone().detach().cpu().numpy()   #TODO: Fix this.
                self.episode_sums[key][env_id] = 0.
            self.infos[env_id]["episode"]["l"] = self.episode_length_buf[env_id].clone().detach().cpu().numpy()
            # log additional curriculum info
            if self.cfg.terrain.curriculum:
                self.infos[env_id]["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            if self.cfg.commands.curriculum:
                self.infos[env_id]["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            # send timeout info to the algorithm
            if self.cfg.env.send_timeouts:
                self.infos[env_id]["time_outs"] = self.time_out_buf
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_torques[env_ids] = 0.  # for RMA
        self.last_contact_forces[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, rews, dones, infos = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        # because SB3 cannot take as input tensors - observation (torch.Tensor) needs to be converted into (np.ndarray)
        obs = obs.detach().cpu().numpy()

        return obs

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def env_is_wrapped(self):
        pass

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf
