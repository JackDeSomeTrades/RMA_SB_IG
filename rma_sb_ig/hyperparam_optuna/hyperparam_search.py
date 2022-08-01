#%% imports

import sys
import os
from rma_sb_ig.models import rma
from rma_sb_ig.utils.helpers import get_config
from box import Box
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
src = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
if not (src in sys.path):
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from envs import create_task
import argparse
import mysql.connector
import optuna
import torch
import numpy as np
from copy import deepcopy
from torch.multiprocessing import Process, set_start_method
from torch import nn as nn
from optuna.samplers import TPESampler

from common.utils import save_dict, create_directory, create_agent
from hyperparam_optuna.hyperparam_sampler import sample_hyperparams
from train_hyperparameter import eval_model

ERRORS_HYPERPARAM=[RuntimeError]
ERRORS_HYPERPARAM=tuple(ERRORS_HYPERPARAM)
WORST_REWARD=0

#%%

class Hyperparam_Searcher:

    def __init__(self, agent:str,
                 task_name:str,
                 hyperparam_range:dict,
                 fun_eval_model, params_fun_eval_model:dict,
                 storage_logs:str=None,
                 study:str="study_hyperparam_search_optuna",
                 params_task:dict=dict(num_env=256, num_threads=4),
                 num_trials: int = 1000,
                 default_hyperparams:dict=dict(device="cuda:0"),
                 num_timesteps:int=int(8e4),
                 num_training:int=3,
                 user:str="pierre",
                 sampler=TPESampler,
                 direction:str="maximize",
                 create_study:bool=True):

        self._study_name = study
        self._username=user
        self._sampler = sampler
        self._num_trials = num_trials
        if create_study:
            self.create_study(direction=direction)
        self._task_name=task_name
        self._params_task=params_task
        self._params_task["device"]=default_hyperparams["device"]
        self._num_training=num_training
        self._default_hyperparams=deepcopy(default_hyperparams)
        self._default_hyperparams["agent"]=agent
        self._hyperparam_range=deepcopy(hyperparam_range)
        self._fun_eval_model = fun_eval_model
        self._params_fun_eval_model=deepcopy(params_fun_eval_model)
        self._params_fun_eval_model["params_task"]=self._params_task
        self._params_fun_eval_model["agent"] = agent
        self._num_timesteps = num_timesteps
        self._save_logs = not(storage_logs is None)
        if self._save_logs:
            self._storage_logs = storage_logs
            create_directory(self._storage_logs)

    def create_study(self, direction="maximize")->None:
        mydb = mysql.connector.connect(host="localhost", user=self._username)
        mycursor = mydb.cursor()
        mycursor.execute("DROP DATABASE IF EXISTS " + self._study_name)
        mycursor.execute("CREATE DATABASE " + self._study_name)
        optuna.create_study(study_name=self._study_name, storage="mysql://"+self._username+"@localhost/" + self._study_name, direction=direction, sampler=self._sampler())

    def sample_params(self, trial: optuna.Trial)->dict:
        params = {}
        for s in self._hyperparam_range.keys():
            if s == "categorical":
                for p in self._hyperparam_range[s]:
                    params[p] = trial.suggest_categorical(p, self._hyperparam_range[s][p])
            elif s == "loguniform":
                for p in self._hyperparam_range[s]:
                    params[p] = trial.suggest_loguniform(p, *self._hyperparam_range[s][p])
            elif s == "uniform":
                for p in self._hyperparam_range[s]:
                    params[p] = trial.suggest_uniform(p, *self._hyperparam_range[s][p])
            elif s == "int":
                for p in self._hyperparam_range[s]:
                    params[p] = trial.suggest_int(p, *self._hyperparam_range[s][p])
        return params

    def get_hyperparams(self, hyperparams_optuna:dict):
        kwargs = self._default_hyperparams.copy()
        cfg = get_config('soto_task_rma_conf.yaml')
        arch_config = Box(cfg).arch_config
        arch = rma.Architecture(arch_config=arch_config, device=arch_config.device,encoder=True)
        policy_kwargs = arch.make_architecture()
        # if "log_std_init" in hyperparams_optuna.keys():
        #     kwargs["policy_kwargs"]["log_std_init"]=hyperparams_optuna["log_std_init"]
        #     del hyperparams_optuna["log_std_init"]
        kwargs["policy_kwargs"] = policy_kwargs
        kwargs.update(hyperparams_optuna)
        if self._default_hyperparams["agent"]=="SAC":
            kwargs["gradient_steps"]=kwargs["train_freq"]
        logs = deepcopy(kwargs)
        if "activation_fn" in logs["policy_kwargs"].keys():
            logs["policy_kwargs"]["activation_fn"]={nn.Tanh: "tanh", nn.ReLU: "relu", nn.ELU: "elu", nn.LeakyReLU: "leaky_relu"}[logs["policy_kwargs"]["activation_fn"]]
        del logs["device"]
        return kwargs, logs

    def objective(self, trial: optuna.Trial)->float:
        objective=0.0
        hyperparams_optuna = self.sample_params(trial=trial)
        kwargs, logs=self.get_hyperparams(hyperparams_optuna)
        params_fun_eval_model = deepcopy(self._params_fun_eval_model)
        try:
            for i in range(self._num_training):
                env=create_task(task_name=self._task_name,
                                num_envs=self._params_task["num_env"],
                                num_threads=self._params_task["num_threads"],
                                device=self._params_task["device"],
                                display=False)
                kwargs["env"] = env
                agent=create_agent(hyperparams=kwargs)
                agent.learn(total_timesteps=self._num_timesteps)
                path_agent=self._storage_logs + str(trial.number)+".pt"
                agent.save(path_agent)
                env.close()
                del agent
                torch.cuda.empty_cache()
                params_fun_eval_model["path_agent"]=path_agent
                objective += self._fun_eval_model(**params_fun_eval_model)
                os.remove(path_agent)
            objective=objective/self._num_training
        except ERRORS_HYPERPARAM as e:
            print(e)
            objective = WORST_REWARD
            del agent
        if self._save_logs:
            logs['trial']=trial.number
            logs["objective"]=objective
            save_dict(filename=self._storage_logs + str(trial.number) + ".json", data=logs)
        return objective

    def optimize(self, num_trials:int) -> None:
        study = optuna.load_study(study_name=self._study_name, sampler=self._sampler(), storage="mysql://"+self._username+"@localhost/"+self._study_name)
        try:
            study.optimize(self.objective, n_trials=num_trials)
        except KeyboardInterrupt:
            pass

    def search(self, num_jobs:int=1) -> None:
        jobs = []
        num_trials=self._num_trials//num_jobs
        for j in range(num_jobs):
            args = (num_trials,)
            process = Process(target=self.optimize, args=args, daemon=True)
            process.start()
            jobs.append(process)
        for j in range(num_jobs):
            jobs[j].join()

#%%

def eval_objective(task_name:str, agent:str, path_agent:str, num_episodes:int, params_task:dict)->np.ndarray:
    return np.mean(eval_model(task_name=task_name, agent=agent, path_agent=path_agent, num_episodes=num_episodes, params_task=params_task))

#%%

NUM_TRIALS=1000
NUM_JOBS=1
NUM_TIMESTEPS=500000
NUM_EVAL_EP=128
NUM_TRAININGS=1
NUM_ENV=128
NUM_THREADS=4
GPU=0

#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task Name.", default='soto')
    parser.add_argument("--agent", type=str, help="Agent (SAC, PPO or DDPG).",default='PPO')
    parser.add_argument("-t", type=int,
                        help="Number of trainings. The default is " + str(NUM_TRAININGS) + ".",
                        default=NUM_TRAININGS)
    parser.add_argument("-n", type=int,
                        help="Number of trials. The default is " + str(NUM_TRIALS) + ".",
                        default=NUM_TRIALS)
    parser.add_argument("--timesteps", type=int,
                        help="Number of timesteps to train the agent. The default is " + str(NUM_TIMESTEPS) + ".",
                        default=NUM_TIMESTEPS)
    parser.add_argument("--num_env", type=int,
                        help="Number of environments per learning. The default is " + str(NUM_ENV) + ".",
                        default=NUM_ENV)
    parser.add_argument("--num_threads", type=int,
                        help="Number of threads for generating environments. The default is " + str(NUM_THREADS) + ".",
                        default=NUM_THREADS)
    parser.add_argument("--eval", type=int,
                        help="Number of episodes to evaluate the algorithm. The default is " + str(NUM_EVAL_EP) + ".", default=NUM_EVAL_EP)
    parser.add_argument("-j", type=int,
                        help="Number of jobs used. The default is " + str(NUM_JOBS) + ".",
                        default=NUM_JOBS)
    parser.add_argument("--multigpu", action="store_true", help="Load the study saved by mysqld for multi-gpu.")
    parser.add_argument("--gpu", type=int,
                        help="GPU used. The default is " + str(GPU) + ".",
                        default=GPU)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    set_start_method('spawn', force=True)


    study = "study_"+ args.agent+ "_"+args.task.replace("-", "_")

    default_hyperparams, hyperparam_range = sample_hyperparams(agent=args.agent, device="cuda:" + str(args.gpu))
    params_fun_eval_model=dict(task_name=args.task, num_episodes=args.eval)
    params_task=dict(num_env=args.num_env, num_threads=args.num_threads)
    searcher = Hyperparam_Searcher(agent=args.agent,
                                   task_name=args.task,
                                   num_timesteps=args.timesteps,
                                   hyperparam_range=hyperparam_range,
                                   default_hyperparams=default_hyperparams,
                                   fun_eval_model=eval_objective,
                                   params_fun_eval_model=params_fun_eval_model,
                                   params_task=params_task,
                                   num_trials=args.n,
                                   num_training=args.t,
                                   storage_logs=study+"/",
                                   study=study,
                                   create_study=not(args.multigpu))

    searcher.search(num_jobs=args.j)


#python hyperparam_search.py Ant PPO -n 100 --timesteps 100 --num_env 10 --eval 10 -j 1



