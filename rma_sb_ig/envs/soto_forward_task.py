import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import time
import torch
CUDA_LAUNCH_BLOCKING=1
# import torch
# from torch import Tensor
# from typing import Tuple, Dict
from rma_sb_ig.utils.helpers import *
from rma_sb_ig.envs.base_task import BaseTask
from rma_sb_ig.utils.soto_scene import SotoEnvScene

import gc
class SotoForwardTask(SotoEnvScene, BaseTask):
    def __init__(self, *args):
        config = args[0][0]
        sim_params = args[0][1]
        args = args[0][2]
        self.compute_rma = not args.avoid_rma
        print("rma is computed : ",  self.compute_rma)
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        BaseTask.__init__(self, cfg=config, sim_params=sim_params, sim_device=args.sim_device)
        SotoEnvScene.__init__(self, cfg=config, physics_engine=args.physics_engine, sim_device=args.sim_device,headless=args.headless, sim_params=sim_params)
        set_seed(config.seed)
        self.l = [[],[],[],[]]
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = self.cfg.rewards.scales.to_dict()
        self.command_angle = np.pi/2
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        if not self.headless :
            self.set_camera()
        self._elapsed_steps = None
        self.observation_space = self._init_observation_space()
        self.action_space = self._init_action_space()

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rb_state = gymtorch.wrap_tensor(_rb_states)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)


        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.rigid_body_tensor = self.rb_state.view(self.num_envs, -1, 13)
        self.soto_root_state = self.root_states.view(self.num_envs,2,13)[:,0,:]
        self.box_root_state = self.root_states.view(self.num_envs,2,13)[:,1,:]
        # initialize some data used later on
        self.common_step_counter = 0
        self.infos = [{} for _ in range(self.num_envs)]
        self.gravity_vec = to_torch(get_axis_params(-1., 2), device=self.device).repeat((self.num_envs, 1))

        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))  # TODO : understand
        self.yaw_vec = to_torch([0., 0., 1.], device=self.device).repeat((self.num_envs, 1))

        self.torques_forces = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)

        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)

        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        self.angle_error = torch.zeros(self.num_envs,device=self.device,dtype=torch.float)
        self.angle_error[:] = torch.pi/2
        self.last_torques_forces = torch.zeros_like(self.torques_forces)  # new addition for RMA
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
         # new addition for RMA
        self.distance_sensors = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.soto_init_state = self.soto_root_state.clone()
        self.box_init_state = self.box_root_state.clone()

        self.box_quat = self.box_root_state[:, 3:7]
        self.box_rpy = get_euler_xyz(self.box_quat)
        self.box_angle = self.box_rpy[2]
        self.box_init_angle = self.box_angle[0]
        self.box_lin_vel = self.box_root_state[..., 7:10]
        self.box_pos = self.box_root_state[..., 0:3]
        self.box_init_pos = self.box_pos.clone()
        self.get_depth_sensors()

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands[:, 0] = self.command_angle
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel],
                                           device=self.device, requires_grad=False)


        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name :
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = self.cfg.control.stiffness['joint']
                self.d_gains[i] = self.cfg.control.damping['joint']
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to joint default values")

        # initialise reward functions here

        self._prepare_reward_function()
        self.init_done = True


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        flag = 0
        # print("actions", actions)

        if type(actions) == np.ndarray:
            # For SB3 compatibility
            flag = 1
            actions = torch.tensor(actions, device=self.device)

        self.actions = torch.clip(actions, self.lower_bounds_joint_tensor, self.upper_bounds_joint_tensor).to(
            self.device)

        # print(self.actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques_forces = self._compute_torques_forces(self.actions).view(self.torques_forces.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques_forces))
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
            #print("This is what's causing the slowdown")
            rew_buf = self.rew_buf.detach().cpu().numpy()
            obs_buf = self.obs_buf.detach().cpu().numpy()
            dones = self.reset_buf.detach().cpu().numpy()
        else:
            rew_buf = self.rew_buf
            obs_buf = self.obs_buf
            dones = self.reset_buf
        torch.cuda.empty_cache()
        #print(len(gc.get_objects()))
        return obs_buf, rew_buf, dones, self.infos




    def _compute_torques_forces(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques_forces = self.p_gains * (
                        actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques_forces = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                        self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques_forces = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        return torch.clip(torques_forces, -self.torque_force_bound,
                          self.torque_force_bound)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.box_quat = self.box_root_state[:, 3:7]

        self.box_lin_vel[:] = self.box_root_state[..., 7:10]

        self.box_pos[:] = self.box_root_state[..., 0:3]
        self.box_rpy = get_euler_xyz(self.box_quat)
        self.box_angle = self.box_rpy[2]
        if self.cfg.commands.heading_command:
            self.commands[:,0] = self.command_angle

        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_torques_forces[:] = self.torques_forces[:]  # for RMA
        self.last_contact_forces[:] = self.contact_forces[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        #     self._draw_debug_vis()


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
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.mean_terminated = torch.mean(self.rew_buf[env_ids]).detach().cpu().numpy()
        env_ids_list = env_ids.tolist()
        for env_id in env_ids_list:
            self.infos[env_id]["episode"] = {}
            for key in self.episode_sums.keys():
                self.infos[env_id]["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_id]) / self.max_episode_length_s
                self.infos[env_id]["episode"]["r"] = self.rew_buf.clone().detach().cpu().numpy()   #TODO: Fix this.
                self.episode_sums[key][env_id] = 0.

            self.infos[env_id]["episode"]["l"] = self.episode_length_buf[env_id].clone().detach().cpu().numpy()
            if self.cfg.commands.curriculum:
                self.infos[env_id]["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            # send timeout info to the algorithm
            if self.cfg.env.send_timeouts:
                self.infos[env_id]["time_outs"] = self.time_out_buf
            self.infos[env_id]["TimeLimit.truncated"] = 1  #TODO: New addition, not checked
            self.infos[env_id]["terminal_observation"] = self.obs_buf[env_id,:].detach().cpu().numpy()
        # reset buffers

        self.last_actions[env_ids] = 0.
        self.last_torques_forces[env_ids] = 0.  # for RMA
        self.last_contact_forces[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] #* torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        self.env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.box_root_state[env_ids] = self.box_init_state[env_ids] #TODO : add noise on this
        self.soto_root_state[env_ids] = self.soto_init_state[env_ids]
        self.env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = self.command_angle



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

    def check_termination(self):
        """ Check if environments need to be reset. Sets up the dones for the return values of step.
        """
        #self.reset_indices = torch.logical_or(self.box_pos[:, 2] < 0.70,self.box_pos[:, 2] > 3)

        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,dim = 1)
        d = self.box_dim[:,0]*torch.sin(self.angle_error) + self.box_dim[:,1]*torch.cos(self.angle_error)
        self.test_pos = torch.logical_and(self.distance_sensors[:,0]+d > 1.3, self.distance_sensors[:,1]+d > 1.3)

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.test_pos
        #self.reset_buf |= self.env_done


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            # else:
            #     self.reward_scales[key] *= self.dt
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
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        # because SB3 cannot take as input tensors - observation (torch.Tensor) needs to be converted into (np.ndarray)
        obs = obs.detach().cpu().numpy()

        return obs

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        pass

    @abstractmethod
    def _init_observation_space(self):
        pass

    @abstractmethod
    def _init_action_space(self):
        pass

    @abstractmethod
    def get_depth_sensors(self):
        pass

    @abstractmethod
    def _get_noise_scale_vec(self, cfg):
        pass

    @abstractmethod
    def compute_observations(self):
        pass
