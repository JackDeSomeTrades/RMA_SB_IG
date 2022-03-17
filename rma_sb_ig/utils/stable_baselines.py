from rma_sb_ig.envs.a1task_rma import A1LeggedRobotTask
from rma_sb_ig.envs.a1_rma_minimal import A1LeggedRobotTaskMinimal
from stable_baselines3.common.vec_env import VecEnv
from abc import abstractmethod
from abc import ABC
from stable_baselines3.common.callbacks import EventCallback, EvalCallback



class SaveHistoryCallback(EventCallback):
    def __init__(self, verbose=0):
        super(SaveHistoryCallback, self).__init__(verbose=verbose)

    def _on_step(self) -> bool:
        zt = self.model.policy.features_extractor.zt
        current_state = self.model.env.X_t
        current_actions = self.model.actions
        print("here")
        return True


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


class RMAA1TaskVecEnvStableBaselineGymMinimal(A1LeggedRobotTaskMinimal, StableBaselinesVecEnvAdapter):
    def __init__(self, *args, **kwargs):
        A1LeggedRobotTaskMinimal.__init__(self, *args, **kwargs)
