import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
src=os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
if not(src in sys.path):
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from train import train_agent

import argparse
import csv
import torch
import pandas as pd
import numpy as np
from torch.multiprocessing import set_start_method
from copy import deepcopy

from common.utils import get_parent_folder, create_directory, collect_files, load_dict
from common.utils import load_hyperparams, save_dict

#%%

def get_hyperparams_csv(path:str)->dict:
    hyperparams_csv=load_dict(path)
    policy_kwargs=deepcopy(hyperparams_csv["policy_kwargs"])
    del hyperparams_csv["policy_kwargs"]
    if 'activation_fn' in policy_kwargs.keys():
        hyperparams_csv["activation_fn"] = policy_kwargs["activation_fn"]
    net_arch=policy_kwargs["net_arch"]
    for i in range(len(net_arch)):
        hyperparams_csv["num_neurons_"+str(i)]=net_arch[i]
    return hyperparams_csv

#%%

def get_hyperparams_from_csv(hyperparams_csv:dict)->dict:
    hyperparams=deepcopy(hyperparams_csv)
    hyperparams_int = ["batch_size", "target_update_interval", "buffer_size", "learning_starts", "train_freq", "gradient_steps", "n_steps", "n_epochs"]
    for h in hyperparams_int:
        if h in hyperparams.keys():
            hyperparams[h]=int(hyperparams[h])
    del hyperparams["objective"], hyperparams["trial"]
    layer_id, net_arch=[], []
    list_hyperparams=list(hyperparams.keys())
    for k in list_hyperparams:
        if "num_neurons" in k:
            layer_id.append(int(k[k.rfind("_") + 1:]))
            net_arch.append(int(hyperparams[k]))
            del hyperparams[k]
    net_arch = np.array(net_arch)[np.argsort(layer_id)]
    policy_kwargs=dict(net_arch=net_arch)
    if 'activation_fn' in hyperparams.keys():
        policy_kwargs["activation_fn"] = hyperparams["activation_fn"]
        del hyperparams["activation_fn"]
    hyperparams["policy_kwargs"]=policy_kwargs
    return hyperparams


#%%

def save_history(path:str, trial_max:int=None)->str:
    json_files=collect_files(path, format=".json")
    if trial_max is None:
        trial_max=len(json_files)
    history=[{}]*trial_max
    if json_files!=[]:
        for f in json_files:
            logs=get_hyperparams_csv(f)
            history[logs["trial"]]=deepcopy(logs)
            field_names=list(logs.keys())
        with open(path + 'history.csv', 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(history)
        return path + 'history.csv'

#%%

def get_elite_hyperparams(path:str, num_elite_hyperparams:int=15)->dict:
    path_elite=get_parent_folder(path)+"elite_hyperparams/"
    create_directory(path_elite)
    df = pd.read_csv(path)
    elite_hyperparams=df.sort_values(by=["objective"], ascending=False)
    elite_hyperparams=elite_hyperparams.head(num_elite_hyperparams)
    elite_trial=elite_hyperparams['trial'].index
    path_elite_hyperparams={}
    for i in range(num_elite_hyperparams):
        hyperparams = get_hyperparams_from_csv(dict(df.iloc[elite_trial[i]]))
        path_elite_hyperparams[elite_trial[i]]=path_elite+'trial_'+str(elite_trial[i]) + '.json'
        save_dict(path_elite_hyperparams[elite_trial[i]], hyperparams)
    return path_elite_hyperparams

#%%

def run_elite_hyperparams(path:str,
                          path_elite_hyperparams:dict,
                          task_name:str,
                          params_task:dict,
                          params_learn:dict,
                          params_callback:dict,
                          num_trainings:int=5,
                          num_jobs:int=5,
                          verbose:bool=True,
                          device:str="cuda:0")->None:

    create_directory(path)
    elite_trials=list(path_elite_hyperparams.keys())
    for trial in elite_trials:
        if verbose:
            print("Computing performance for trial "+str(trial)+"...")
        path_trial = path + "trial_" + str(trial) + "/"
        hyperparams = load_hyperparams(path_elite_hyperparams[trial], device=device)
        train_agent(path=path_trial, task_name=task_name, params_task=params_task, hyperparams=hyperparams, params_learn=params_learn,
                    params_callback=params_callback, num_training=num_trainings, num_jobs=num_jobs, verbose=False)


#%%

NUM_ELITE_HYPERPARAMS=15
NUM_TIMESTEPS=1200000
NUM_TRAININGS=5
NUM_EVAL_EP=100
NUM_ENV=4096
NUM_THREADS=4
NUM_JOBS=1
GPU=0

#%%

if __name__ == "__main__":
    # save_history("study_SAC_Ant/json/", trial_max=463)

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path where the logs are saved.")
    parser.add_argument("task", type=str, help="Task Name.")
    parser.add_argument("-f", type=int, help="Evaluate the model every f steps.")
    parser.add_argument("-e", type=int,
                        help="Select the e best hyperparameter settings. The default is" + str(NUM_ELITE_HYPERPARAMS) + ".", default=NUM_ELITE_HYPERPARAMS)
    parser.add_argument("-t", type=int,
                        help="Number of timesteps to train the agent. The default is " + str(NUM_TIMESTEPS) + ".", default=NUM_TIMESTEPS)
    parser.add_argument("-n", type=int,
                        help="Number of trainings to evaluate each hyperparameter. The default is "+str(NUM_TRAININGS)+".", default=NUM_TRAININGS)
    parser.add_argument("--eval", type=int,
                        help="Number of episodes to evaluate the agent. The default is " + str(NUM_EVAL_EP) + ".", default=NUM_EVAL_EP)
    parser.add_argument("--num_env", type=int,
                        help="Number of environments per learning. The default is " + str(NUM_ENV) + ".",
                        default=NUM_ENV)
    parser.add_argument("--num_threads", type=int,
                        help="Number of threads for generating environments. The default is " + str(NUM_THREADS) + ".",
                        default=NUM_THREADS)
    parser.add_argument("--verbose", action="store_true",
                        help="Display messages.")
    parser.add_argument("-j", type=int,
                        help="Number of jobs used. The default is " + str(NUM_JOBS) + ".", default=NUM_JOBS)
    parser.add_argument("--gpu", type=int,
                        help="GPU used. The default is " + str(GPU) + ".", default=GPU)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    set_start_method('spawn', force=True)

    path_elite_hyperparams=get_elite_hyperparams(args.csv, num_elite_hyperparams=args.e)
    params_task = dict(num_env=args.num_env, num_threads=args.num_threads)
    params_learn = dict(total_timesteps=args.t)
    params_callback = dict(num_eval_episodes=args.eval, freq=args.f, save_agent=False)


    run_elite_hyperparams(path=get_parent_folder(args.csv),
                          path_elite_hyperparams=path_elite_hyperparams,
                          task_name=args.task,
                          params_task=params_task,
                          params_learn=params_learn,
                          params_callback=params_callback,
                          num_trainings=args.n,
                          num_jobs=args.j,
                          verbose=args.verbose,
                          device="cuda:"+str(args.gpu))

