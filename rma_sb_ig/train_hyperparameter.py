from envs import create_task

import os
import torch
import argparse
import numpy as np
from torch.multiprocessing import Process, set_start_method
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC, PPO, DDPG

from common.utils import CloudpickleWrapper
from common.utils import create_directory
from common.utils import create_agent, load_hyperparams
from common.utils import collect_files
from plot_perf import plot_eval_curve


#%%

NUM_TIMESTEPS=500000
NUM_TRAININGS=10
NUM_EVAL_EPISODES=20
NUM_ENV=100
NUM_THREADS=4
NUM_JOBS=1
GPU=0
DEVICE=torch.device(torch.cuda.current_device())

#%%

def eval_model(task_name:str, agent:str, path_agent:str, num_episodes:int, params_task:dict)->np.ndarray:
    rollout_reward = []
    num_env=params_task["num_env"]
    if num_episodes<num_env:
        num_env=num_episodes
    eval_env = create_task(task_name=task_name,
                      num_envs=num_env,
                      num_threads=params_task["num_threads"],
                      device=params_task["device"],
                      display=False)
    type_models = dict(SAC=SAC, DDPG=DDPG, PPO=PPO)
    model = type_models[agent].load(path_agent, env=eval_env, device=params_task["device"])
    episode_counts = np.zeros(num_env, dtype="int")
    episode_count_targets = np.array([(num_episodes + i) // num_env for i in range(num_env)], dtype="int")
    current_rewards = np.zeros(num_env)
    observations = eval_env.reset()
    while (episode_counts < episode_count_targets).any():
        actions, _ = model.predict(observations, deterministic=True)
        observations, rewards, dones, _ = eval_env.step(actions)
        current_rewards += rewards
        for i in range(num_env):
            if episode_counts[i] < episode_count_targets[i]:
                if dones[i]:
                    rollout_reward.append(current_rewards[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
    eval_env.close()
    del model
    torch.cuda.empty_cache()
    return np.array(rollout_reward)

#%%

def eval_perf(path:str, agent:str, task_name:str, params_task:dict, num_eval_episodes:int, save_agent:bool=False)->None:
    path_agent=collect_files(path=path, format=".pt")
    timesteps=np.array([int(p[p.rfind("_")+1:-3]) for p in path_agent])
    id_timesteps=np.argsort(timesteps)
    evaluation_rewards = []
    for p in path_agent:
        evaluation_rewards.append(eval_model(task_name=task_name, params_task=params_task, agent=agent, path_agent=p, num_episodes=num_eval_episodes))
    evaluation_rewards=np.array(evaluation_rewards)
    np.savez(path[:-1]+".npz",
             timesteps=timesteps[id_timesteps],
             rewards=evaluation_rewards[id_timesteps])
    if not(save_agent):
        for p in path_agent:
            os.remove(p)

#%%

class SaveModelsCallback(BaseCallback):
    def __init__(self, path:str, freq:int=1000, verbose:int=0,):
        super(SaveModelsCallback, self).__init__(verbose)
        self.path=path
        create_directory(path+"/")
        self.freq = freq

    def _on_step(self) -> bool:
        print(self.n_calls, self.freq)
        if self.freq > 0 and self.n_calls % self.freq == 0:
            print(self.path+"model_"+str(self.n_calls)+".pt")
            self.model.save(self.path+"model_"+str(self.n_calls)+".pt")
        return True

#%%

def _train_agent(task_name:str, params_task_fn_wrapper, hyperparams_fn_wrapper, params_learn_fn_wrapper, params_callback_fn_wrapper)->None:
    params_task=params_task_fn_wrapper.var
    hyperparams = hyperparams_fn_wrapper.var
    params_learn=params_learn_fn_wrapper.var
    params_callback = params_callback_fn_wrapper.var
    params_task["device"]=hyperparams["device"]
    num_eval_episodes=20
    if "num_eval_episodes" in params_callback.keys():
        num_eval_episodes=params_callback["num_eval_episodes"]
        del params_callback["num_eval_episodes"]
    save_agent = False
    if "save_agent" in params_callback.keys():
        save_agent=params_callback["save_agent"]
        del params_callback["save_agent"]
    env = create_task(task_name=task_name,
                      num_envs=params_task["num_env"],
                      num_threads=params_task["num_threads"],
                      device=params_task["device"],
                      display=False)
    hyperparams["env"]=env
    agent=create_agent(hyperparams=hyperparams)
    if params_callback is not None:
        callback = SaveModelsCallback(**params_callback)
        params_learn["callback"]=callback
    agent.learn(**params_learn)
    env.close()
    del agent
    torch.cuda.empty_cache()
    eval_perf(path=params_callback["path"], agent=hyperparams["agent"], task_name=task_name, params_task=params_task, num_eval_episodes=num_eval_episodes, save_agent=save_agent)

#%%

def train_agent(path:str, task_name:str, params_task:dict, hyperparams:dict, params_learn:dict, params_callback:dict, num_training:int=1, num_jobs:int=1, verbose:bool=True)->None:
    create_directory(path)
    n_iter_full_jobs = num_training // num_jobs
    n_remaining_jobs = num_training % num_jobs
    n_runs = n_iter_full_jobs + (num_training % num_jobs != 0)
    def _launch_jobs(n_processes, n_run):
        processes = []
        for j in range(n_processes):
            params_callback["path"]=path+"training_"+str(j+n_run*num_jobs)+"/"
            args = (task_name, CloudpickleWrapper(params_task), CloudpickleWrapper(hyperparams), CloudpickleWrapper(params_learn), CloudpickleWrapper(params_callback))
            process = Process(target=_train_agent, args=args, daemon=True)
            process.start()
            processes.append(process)
        for j in range(n_processes):
            processes[j].join()
    if verbose:
        print("Training...")
    for i in range(n_iter_full_jobs):
        if verbose:
            print("\rRun {}/{}".format(i+1, n_runs), end="")
        _launch_jobs(num_jobs, i)
    if n_remaining_jobs!=0:
        if verbose:
            print("\rRun {}/{}".format(i+1, n_runs), end="")
        _launch_jobs(n_remaining_jobs, n_iter_full_jobs)
    plot_eval_curve(path)

#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task Name.")
    parser.add_argument("hyperparams", type=str, help="Json file containing hyperparameters.")
    parser.add_argument("-p", type=str, help="Path where to save logs.", default="")
    parser.add_argument("-f", type=int, help="Evaluate the model every f steps.")
    parser.add_argument("-t", type=int,
                        help="Number of timesteps to train the agent. The default is " + str(NUM_TIMESTEPS) + ".", default=NUM_TIMESTEPS)
    parser.add_argument("-n", type=int,
                        help="Number of trainings to evaluate the agent. The default is "+str(NUM_TRAININGS)+".", default=NUM_TRAININGS)
    parser.add_argument("--eval", type=int,
                        help="Number of episodes to evaluate the algo. The default is " + str(NUM_EVAL_EPISODES) + ".", default=NUM_EVAL_EPISODES)
    parser.add_argument("--num_env", type=int,
                        help="Number of environments per learning. The default is " + str(NUM_ENV) + ".",
                        default=NUM_ENV)
    parser.add_argument("--num_threads", type=int,
                        help="Number of threads for generating environments. The default is " + str(NUM_THREADS) + ".",
                        default=NUM_THREADS)
    parser.add_argument("-j", type=int,
                        help="Number of jobs used. The default is " + str(NUM_JOBS) + ".", default=NUM_JOBS)
    parser.add_argument("--verbose", action="store_true",
                        help="Display messages.")
    parser.add_argument("--save_agent", action="store_true",
                        help="Save agents.")
    parser.add_argument("--gpu", type=int,
                        help="GPU used. The default is " + str(GPU) + ".", default=GPU)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    #python hyperparam_search.py Ant PPO -n 100 --timesteps 100 --num_env 10 --eval 10 -j 1

    hyperparams=load_hyperparams(path=args.hyperparams, device="cuda"+str(args.gpu))

    path_logs = args.p + args.task + "-"+hyperparams["agent"]+"/"
    params_learn = dict(total_timesteps=args.t)
    params_callback = dict(freq=args.f, num_eval_episodes=args.eval, save_agent=args.save_agent)
    params_task = dict(num_env=args.num_env, num_threads=args.num_threads)

    set_start_method('spawn', force=True)
    train_agent(path=path_logs, task_name=args.task, params_task=params_task, hyperparams=hyperparams, params_learn=params_learn, params_callback=params_callback,
              num_training=args.n, num_jobs=args.j, verbose=args.verbose)

