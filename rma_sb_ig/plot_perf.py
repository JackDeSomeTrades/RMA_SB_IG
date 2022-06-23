#%% imports

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd

from copy import deepcopy

sns.set_theme(style="darkgrid")
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

from common.utils import get_parent_folder
from common.utils import collect_files

#%%

def smoothen(x:np.ndarray, winsize:int=5):
    return np.array(pd.Series(x).rolling(winsize).mean())[winsize-1:]

#%%

def extract_rewards(path_logs:str)->dict:
    training_logs = collect_files(path=path_logs, format=".npz")
    rewards_logs = []
    for t in training_logs:
        logs = np.load(t)
        timesteps, rewards = logs["timesteps"], np.mean(logs["rewards"], axis=1)
        rewards_logs.append(rewards)
    conf_level = 0.95
    num_trainings = len(training_logs)
    rewards_logs = np.array(rewards_logs)
    avg_rewards = np.mean(rewards_logs, axis=0)
    conf_rewards = np.std(rewards_logs, axis=0)/np.sqrt(num_trainings)*stats.t.ppf(1-(1-conf_level)/2, num_trainings-1)
    return dict(timesteps=timesteps, avg=avg_rewards, conf=conf_rewards)

#%%

def plot_eval_curve(path_logs:str)->None:
    rewards=extract_rewards(path_logs)
    plt.figure()
    plt.plot(rewards["timesteps"], rewards["avg"])
    plt.fill_between(rewards["timesteps"], rewards["avg"]-rewards["conf"], rewards["avg"]+rewards["conf"], alpha=0.1)
    plt.xlabel("Timesteps")
    plt.ylabel("Average Return Per Episode")
    plt.title("Evaluation Curve", fontweight="bold")
    plt.savefig(path_logs + "eval_curve.png")


#%%

def plot_eval_curve_all(path_logs:dict, path:str="",
                        title:str="Evaluation Curves", perf_opt:float=None, winsize:int=None, timesteps_max:int=None)->None:
    labels=list(path_logs.keys())
    timesteps, avg_rewards, conf_rewards={}, {}, {}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for l in labels:
        rewards=extract_rewards(path_logs[l])
        timesteps[l], avg_rewards[l], conf_rewards[l] = rewards["timesteps"], rewards["avg"], rewards["conf"]
    if not(winsize is None):
        for l in labels:
            avg_rewards[l] = smoothen(avg_rewards[l], winsize=winsize)
            conf_rewards[l] = smoothen(conf_rewards[l], winsize=winsize)
            timesteps[l] = timesteps[l][:len(avg_rewards[l])]
    plt.figure(figsize=(12,8))
    if not(perf_opt is None):
        ep=np.copy(timesteps[labels[-1]])
        ep[0], ep[-1]=ep[0]-1000, ep[-1]+1000
        plt.plot(ep, np.ones((len(timesteps[labels[-1]])))*perf_opt, label="Optimal Performance", linestyle="--", color=colors[0])
    for i in range(len(labels)):
        plt.plot(timesteps[labels[i]], avg_rewards[labels[i]], label=labels[i])#, color=colors[i+1])
        plt.fill_between(timesteps[labels[i]], avg_rewards[labels[i]]-conf_rewards[labels[i]], avg_rewards[labels[i]]+conf_rewards[labels[i]], alpha=0.1)#, color=colors[i+1])
    plt.xlabel("Timesteps")
    if timesteps_max is None:
        timesteps_max=timesteps[l][-1]
    plt.xlim((timesteps[l][0]-100, timesteps_max))
    plt.ylabel("Average Return Per Episode")
    plt.legend()
    plt.title(title, fontweight="bold")
    print(path+title.replace(" ", "_"))
    plt.savefig(path+title.replace(" ", "_")+".png")

#%%

def plot_elite_eval_curves(path_elite_trials:str, perf_opt:float=None, winsize:int = None, timesteps_max:int=None, trial_selected:list=None):
    trial_direc=[t for t in os.listdir(path_elite_trials) if t.rfind("trial_") != -1]
    trial={t.replace("_", " "):os.path.join(path_elite_trials, t) for t in trial_direc}
    if not(trial_selected is None):
        tmp=deepcopy(trial)
        trial={}
        for t in trial_selected:
            trial["trial "+str(t)]=tmp["trial "+str(t)]
    plot_eval_curve_all(path_logs=trial, path=path_elite_trials, title= "Evaluation Curves of Elite Hyperparameter Setting", perf_opt=perf_opt, winsize=winsize, timesteps_max=timesteps_max)

#%%

def get_rank_elite_hyperparams(path_elite_trials:str):
    trial_direc = [t for t in os.listdir(path_elite_trials) if t.rfind("trial_") != -1]
    trial = {t.replace("_", " "): os.path.join(path_elite_trials, t) for t in trial_direc}
    labels = np.array(list(trial.keys()))
    num_trials=len(labels)
    perf, avg_rewards, conf_rewards=np.zeros((num_trials,)), np.zeros((num_trials,)), np.zeros((num_trials,))
    for i in range(num_trials):
        print(trial[labels[i]])
        rewards=extract_rewards(trial[labels[i]])
        avg_rewards[i], conf_rewards[i] = rewards["avg"][-1], rewards["conf"][-1]
        perf[i]=avg_rewards[i]-conf_rewards[i]
    id_trial=np.argsort(perf)
    labels, perf, avg_rewards, conf_rewards=labels[id_trial], perf[id_trial], avg_rewards[id_trial], conf_rewards[id_trial]
    for i in range(num_trials):
        print(str(num_trials-i)+" : "+str(labels[i])+" - "+str(perf[i])+" - "+str(avg_rewards[i])+" - "+str(conf_rewards[i])+" - ")

#%%

def plot_param_hyperparam_search(path:str, param:str, save_fig:bool=False, show:bool=True, log_scale:bool=False)->None:
    df = pd.read_csv(path)
    trial_hyperparams=df.sort_values(by=["trial"])
    param_trial=np.array(trial_hyperparams[param])
    trial=np.array(trial_hyperparams["trial"])
    elite_hyperparams=df.sort_values(by=[param])
    param_objective=np.array(elite_hyperparams[param])
    objective=np.array(elite_hyperparams["objective"])
    param_objective_u=np.unique(param_objective)
    num_params=len(param_objective_u)
    objective_u=np.zeros((num_params))
    for i in range(num_params):
        objective_u[i]=np.mean(objective[param_objective==param_objective_u[i]])
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.scatter(trial, param_trial)
    plt.xlabel("Trial")
    plt.ylabel(param+" Value Tested")
    if log_scale:
        plt.yscale("log")
    plt.subplot(1,2,2)
    plt.plot(param_objective_u, objective_u)
    plt.ylabel("Objective")
    plt.xlabel(param+" Value Tested")
    if log_scale:
        plt.xscale("log")
    plt.suptitle("Hyperparameter Search on "+str(param)+" with Optuna", fontweight="bold")
    if save_fig:
        plt.savefig(get_parent_folder(path)+param+"_hyperparam_search.png")
    if show:
        plt.show()
