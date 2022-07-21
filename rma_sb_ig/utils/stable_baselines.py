from rma_sb_ig.envs.a1task_rma import A1LeggedRobotTask
from rma_sb_ig.envs.v0_task_rma import V0LeggedRobotTask
from rma_sb_ig.envs.v0six_task_rma import V0SixLeggedRobotTask
from rma_sb_ig.envs.Sototask_rma import SotoRobotTask
from stable_baselines3.common.vec_env import VecEnv
from abc import abstractmethod
from abc import ABC
from stable_baselines3.common.callbacks import EventCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np
import hickle as hkl
import os
from pathlib import Path
from rma_sb_ig.utils import *
import pandas as pd
from tqdm import tqdm
import torch
class SaveHistoryCallback(EventCallback):
    def __init__(self, savepath=None, verbose=0):
        super(SaveHistoryCallback, self).__init__(verbose=verbose)
        if savepath is not None:
            svpathparent = Path(savepath).parent
            if os.path.exists(svpathparent) is False:
                Path(savepath).mkdir(parents=True, exist_ok=True)
            self.savepath = savepath
        else:
            raise ValueError("Provide a path to save environment data")
        self.file = open(self.savepath, 'w')
        self.datadict = {}
        self.data_set = pd.DataFrame()

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        zt = self.model.policy.features_extractor.zt.clone().detach().cpu().numpy()
        current_state = self.model.env.X_t.clone().detach().cpu().numpy()
        current_actions = self.model.env.actions.clone().detach().cpu().numpy()
        self.datadict[self.n_calls] = {
            'state': current_state, 'env_encoding': zt, 'actions': current_actions}
        # Log scalar value (here a random variable)
        return True

    def _on_training_end(self) -> None:
        if self.model.env.final_computation :
            max_len_datadict = max(self.datadict.keys())
            for step in tqdm(range(1,max_len_datadict),desc="computing observation through encoder") :
                state_at_step = self.datadict[step]['state']
                state = torch.tensor(state_at_step,device ="cuda",dtype=torch.float)
                out = self.model.policy.extract_features(state)
                self.datadict[step]['env_encoding'] = out.clone().detach().cpu().numpy()
            hkl.dump(self.datadict, self.file)
            self.file.close()
        


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


@register("a1")
class RMAA1TaskVecEnvStableBaselineGym(A1LeggedRobotTask, StableBaselinesVecEnvAdapter):
    def __init__(self, *args, **kwargs):
        A1LeggedRobotTask.__init__(self, *args, **kwargs)


@register("v0")
class RMAV0TaskVecEnvStableBaselineGym(V0LeggedRobotTask, StableBaselinesVecEnvAdapter):
    def __init__(self, *args, **kwargs):
        V0LeggedRobotTask.__init__(self, *args, **kwargs)


@register("v0Six")
class RMAV0SixTaskVecEnvStableBaselineGym(V0SixLeggedRobotTask, StableBaselinesVecEnvAdapter):
    def __init__(self, *args, **kwargs):
        V0SixLeggedRobotTask.__init__(self, *args, **kwargs)


@register("soto")
class RMASotoTaskVecEnvStableBaseLineGym(SotoRobotTask, StableBaselinesVecEnvAdapter):
    def __init__(self, *args, **kwargs):  # args : tuple / kwargs : dictionnary

        SotoRobotTask.__init__(self, *args, **kwargs)
