import sys
import os
from rma_sb_ig.utils.helpers import get_config, parse_config
from rma_sb_ig.utils import env_gen
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
src = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
if not (src in sys.path):
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import pathlib
import yaml

def create_task(task_name:str, num_envs:int=4, num_threads:int=4, device:str="cuda:0", display:bool=False):
    # Mappings from strings to environments

    cfg_filename=str(pathlib.Path(__file__).parent.parent.absolute())+"/cfg/"+"soto_task_rma_conf"+".yaml"

    config = get_config(cfg_filename)

    config["task_config"]["env"]["num_envs"]=num_envs
    config["task_config"]["sim_param"]["physx"]["num_threads"]=num_threads
    config["task_config"]["sim_param"]["physx"]["num_subscenes"]=num_threads
    env = env_gen(task_name)(parse_config(config),final_computation = True
    )

    return env