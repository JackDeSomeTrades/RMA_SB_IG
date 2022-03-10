from rma_sb_ig.envs.a1task_rma import A1LeggedRobotTask
from stable_baselines3.common.vec_env import VecEnv
from abc import abstractmethod
from abc import ABC


class StableBaselinesVecEnvAdapter(VecEnv):

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def seed(self, seed):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        pass

    def close(self):
        pass

    def reset(self):
        pass


class RMAA1TaskVecEnvStableBaselineGym(A1LeggedRobotTask, StableBaselinesVecEnvAdapter):
    def __init__(self, *args, **kwargs):
        A1LeggedRobotTask.__init__(self, *args, **kwargs)
