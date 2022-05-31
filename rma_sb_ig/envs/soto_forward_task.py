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

        # self.dt = self.cfg.control.decimation * self.sim_params.dt
        # self.obs_scales = self.cfg.normalization.obs_scales
        # self.reward_scales = self.cfg.rewards.scales.to_dict()
        # self.command_ranges = self.cfg.commands.ranges.to_dict()
        # if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
        #     self.cfg.terrain.curriculum = False
        # self.max_episode_length_s = self.cfg.env.episode_length_s
        # # in terms of timsesteps.
        # self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # # to count the number of times the step function is called.
        # self._elapsed_steps = None

        # self.cfg.domain_rand.push_interval = np.ceil(
        #     self.cfg.domain_rand.push_interval_s / self.dt)

        self.observation_space = self._init_observation_space()
        self.action_space = self._init_action_space()
        self.init_done = True

        if self.is_test_mode:
            self._process_test()
        else:
            self.gym.destroy_viewer(self.viewer)
            self.gym.destroy_sim(self.sim)

    def _process_test(self):

        self.gym.prepare_sim(self.sim) #TODO : ATTENTION FAIT TOUT PLANTER avec set actor dof state
        # when gym.prepared_sim is called, have to work withoot state tensors : gym.set_actor_root_state_tensor(sim, _root_tensor)
        # acquire root state tensor descriptor
        env_print = 2
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
            # k = np.abs(np.sin(step / 1000))

            # #actualise position
            #
            # self.soto_current = k * self.upper_bounds_joints + \
            #     (1 - k) * self.lower_bounds_joints
            # self.default_dof_pos = self.soto_current
            #
            # # remember : important pieces to control are conveyor belt left base link/conveyor belt right base link
            # self.default_dof_state["pos"] = self.default_dof_pos

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
            self.gym.sync_frame_time(self.sim) #synchronise simulation with real time
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

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
        pass

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
