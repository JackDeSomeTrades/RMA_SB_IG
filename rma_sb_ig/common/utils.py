import numpy as np
import json
import cloudpickle
from torch import nn as nn
from stable_baselines3 import SAC, PPO, DDPG
import os

#%%

class Numpy_Encoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

#%%

class CloudpickleWrapper(object):
    def __init__(self, var):
        """
            Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
            :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = cloudpickle.loads(obs)


#%%

def save_dict(filename:str, data:dict)->None:
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4, sort_keys=True, separators=(', ', ': '), ensure_ascii=False, cls=Numpy_Encoder)

#%%

def load_dict(filename:str)->dict:
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

#%%

def create_directory(path:str)->None:
    if not (os.path.exists(path)):
        try:
            os.makedirs(path)
        except OSError:
            print("Failed to create %s" % path)

#%%

def get_parent_folder(filename:str)->str:
    return filename[0:filename.rfind('/')+1]

#%%

def get_filename(path_filename:str)->str:
    return path_filename[path_filename.rfind('/')+1:]

#%%

def collect_files(path:str, format:str)->list:
    return [os.path.join(path, f) for f in os.listdir(path) if f[-len(format):]==format]

#%%

def delete_directory(path:str)->None:
    for f in os.listdir(path):
        os.remove(path+f)
    os.rmdir(path)

#%%

def load_hyperparams(path:str, device:str="cpu")->dict:
    hyperparams=load_dict(path)
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
    if "activation_fn" in hyperparams["policy_kwargs"]:
        hyperparams["policy_kwargs"]["activation_fn"]=activation_fn[hyperparams["policy_kwargs"]["activation_fn"]]
    hyperparams["device"]=device
    return hyperparams

#%%

def create_agent(hyperparams:dict):
    type_models = dict(SAC=SAC, DDPG=DDPG, PPO=PPO)
    agent=hyperparams["agent"]
    del hyperparams["agent"]
    model=type_models[agent](**hyperparams)
    hyperparams["agent"]=agent
    return model

