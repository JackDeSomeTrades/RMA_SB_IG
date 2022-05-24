from time import time
from warnings import WarningMessage
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
from rma_sb_ig.utils.scene import SotoEnvScene


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

        self.init_done = True
        if self.is_test_mode :
            self._process_test()
        else :
            self.gym.destroy_viewer(self.viewer)
            self.gym.destroy_sim(self.sim)

    def _process_test(self):
        self.gym.prepare_sim(self.sim) #TODO : ATTENTION FAIT TOUT PLANTER avec set actor dof state
        #when gym.prepared_sim is called, have to work withroot state tensors : gym.set_actor_root_state_tensor(sim, _root_tensor)
        # acquire root state tensor descriptor
        step = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            step += 1
            # update viewer
            # set initial dof states
            # set initial position targets #TODO : work only if sim is not prepared
            # for env in self.envs :
            #     # set dof states
            #     self.gym.set_actor_dof_states(
            #         env, self.soto_handle, self.default_dof_state, gymapi.STATE_ALL)
            #
            #     # set position targets
            #     self.gym.set_actor_dof_position_targets(
            #         env, self.soto_handle, self.default_dof_pos)
            # k = np.abs(np.sin(step / 100))
            # self.soto_current = k * self.soto_upper_limits + \
            #                     (1 - k) * self.soto_lower_limits
            # self.default_dof_pos = self.soto_current
            #
            # # remember : important pieces to control are conveyor belt left base link/conveyor belt right base link
            # self.default_dof_state["pos"] = self.default_dof_pos
            #
            # # send to torch
            # self.default_dof_pos_tensor = to_torch(
            #     self.default_dof_pos, device=self.device)

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            #self.gym.sync_frame_time(self.sim) #synchronise simulation with real time
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        # cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)



    def step(self, actions):
        pass

    def reset(self):
        pass

    def get_observations(self):
        pass

    def get_privileged_observations(self):
        pass
