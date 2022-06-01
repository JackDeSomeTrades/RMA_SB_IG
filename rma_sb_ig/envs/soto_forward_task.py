from time import time
from warnings import WarningMessage

import matplotlib.pyplot as plt
import numpy as np
import os
from abc import ABC, abstractmethod

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
from rma_sb_ig.utils.soto_scene import SotoEnvScene


class SotoForwardTask(SotoEnvScene, BaseTask):
    def __init__(self, *args):
        config = args[0][0]
        sim_params = args[0][1]
        args = args[0][2]

        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        BaseTask.__init__(self, cfg=config,
                          sim_params=sim_params, sim_device=args.sim_device)

        SotoEnvScene.__init__(self, cfg=config, physics_engine=args.physics_engine, sim_device=args.sim_device,
                              headless=args.headless, sim_params=sim_params)
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = self.cfg.rewards.scales.to_dict()
        self.command_ranges = self.cfg.commands.ranges.to_dict()
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        # in terms of timsesteps.
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # to count the number of times the step function is called.
        self._elapsed_steps = None

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt)

        self.observation_space = self._init_observation_space()
        self.action_space = self._init_action_space()
        self.init_done = True

        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = self.cfg.rewards.scales.to_dict()

        if self.is_test_mode:
            self._process_test()
        else:
            self.gym.destroy_viewer(self.viewer)
            self.gym.destroy_sim(self.sim)

    def _process_test(self):

        self.gym.prepare_sim(self.sim) #TODO : ATTENTION FAIT TOUT PLANTER avec set actor dof state
        # when gym.prepared_sim is called, have to work withoot state tensors : gym.set_actor_root_state_tensor(sim, _root_tensor)
        # acquire root state tensor descriptor
        step = 0
        distance_from_cameras = [[] for i in range(self.num_envs)]
        for i in range(self.num_envs) :
            distance_from_cameras[i].append([])
            distance_from_cameras[i].append([])
        while True:
            step += 1
            #set initial position targets #TODO : work only if sim is not prepared

            # for env in self.envs:
            #     # set position targets
            #     self.gym.set_actor_dof_states(env,self.soto_handle,self.default_dof_state,gymapi.STATE_ALL)
            #     self.gym.set_actor_dof_position_targets(
            #         env, self.soto_handle, self.default_dof_pos)
            self.gym.sync_frame_time(self.sim)  # synchronise simulation with real time
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # camera
            self.gym.render_all_camera_sensors(self.sim)
            for i in range(self.num_envs) :
                depth_image1 = self.gym.get_camera_image(
                    self.sim, self.envs[i], self.distance_handles[i][0], gymapi.IMAGE_DEPTH)
                depth_image2 = self.gym.get_camera_image(
                    self.sim, self.envs[i], self.distance_handles[i][1], gymapi.IMAGE_DEPTH)
                if depth_image1[0][0] <= - self.cfg.distance_sensor.far_plane :
                    depth_image1 = [[- self.cfg.distance_sensor.far_plane]]
                if depth_image2[0][0] <= - self.cfg.distance_sensor.far_plane :
                    depth_image2 = [[- self.cfg.distance_sensor.far_plane]]

                distance_from_cameras[i][0].append(depth_image1[0][0])
                distance_from_cameras[i][1].append(depth_image2[0][0])

            # print()
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
                if self.gym.query_viewer_has_closed(self.viewer):
                    break



        # cleanup
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        plt.figure(figsize=(20,20))
        for i in range(self.num_envs) :
            plt.subplot(2,3,i+1)
            plt.plot([i for i in range(step)][2:],distance_from_cameras[i][0][2:],'r',label = 'right')
            plt.plot([i for i in range(step)][2:],distance_from_cameras[i][1][2:],'g',label = 'left')
            plt.title("distance capteurs boite environnement {}".format(i))
            plt.legend()
            plt.grid()
        plt.show()


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

        self.actions = torch.clip(actions, self.action_space.low, self.action_space.high).to(self.device) #TODO : verifier si ça fonctionne
        # print(self.actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques_forces = self._compute_torques_forces(self.actions).view(self.torques.shape)

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
            # print("This is what's causing the slowdown")
            rew_buf = self.rew_buf.detach().cpu().numpy()
            obs_buf = self.obs_buf.detach().cpu().numpy()
            dones = self.reset_buf.detach().cpu().numpy()
        else:
            rew_buf = self.rew_buf
            obs_buf = self.obs_buf
            dones = self.reset_buf

        # print("#", "--"*50, "#")

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
        #pd controller
        actions_scaled = actions
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques_forces = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type == "V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques_forces, -self.torque_limits, self.torque_limits)


    def reset(self):
        pass

    def get_observations(self):
        pass

    def get_privileged_observations(self):
        pass

    @abstractmethod
    def _init_observation_space(self):
        pass

    @abstractmethod
    def _init_action_space(self):
        pass

    @abstractmethod
    def _get_noise_scale_vec(self, cfg):
        pass

    @abstractmethod
    def compute_observations(self):
        pass
