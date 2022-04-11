import yaml
import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import quat_apply, normalize
import numpy as np
import random
import torch
import box
import argparse
import datetime

from rma_sb_ig.cfg import CFGFILEPATH
from pathlib import Path

torch.set_printoptions(profile="full")


class UserNamespace(object):
    pass


def exists(namespace):
    try:
        if namespace:
            return True
    except Exception:
        return False


def parse_replay_config(args, cfg):
    config = cfg['task_config']
    sim_params = parse_sim_params(args, config)
    config = box.Box(config)

    return config, sim_params, args


def parse_config(cfg):
    config = cfg['task_config']
    args = get_args()  # needs to be done only to follow gymutils implements it this way. Future work: to redo this.
    sim_params = parse_sim_params(args, config)
    config = box.Box(config)

    return config, sim_params, args


def get_run_name(cfg, args):
    try:
        run_name = cfg.logging.run_name
    except (AttributeError, KeyError):
        idx_ = []
        alg_type = []
        log_dir = cfg.logging.dir.format(ROOT_DIR=Path.cwd())
        folders = os.listdir(log_dir)
        for folder in folders:
            if folder[0] != '.':
                idx_.append(int(folder.split('_')[1]))
                alg_type.append(folder.split('_')[0])

        if len(idx_) == 0:
            idx_ = [0]
            alg_type = ["ppo"]

        max_id = max(idx_)
        conf_keys = list(cfg.keys())
        alg_type = [algs.lower() for algs in alg_type]
        alg_list = [conf_key for conf_key in conf_keys if conf_key.lower() in alg_type]
        # alg_lst = [value for value in alg_type if value in conf_keys]  # should generally return a list of 1 element.

        if (args.run_comment is not None) or args.timestamp:
            run_comment = args.run_comment + '_' + datetime.datetime.now().strftime("%d_%b_%H%M%S") + '_'
        else:
            run_comment = '_'

        run_name = alg_list[0].upper()+'_'+str((max_id+1)) + '_' + run_comment + args.robot_name

    return run_name


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_config(cfg_fname):
    conf_file = os.path.join(CFGFILEPATH, cfg_fname)
    with open(conf_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_args():
    custom_parameters = [
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--replay_cfg", "type": str, "default": "a1_task_rma", "help": "Task config for replaying the trained policy"},
        {"name": "--experiment_name", "type": str, "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str, "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int, "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        if type(cfg["sim"]["physx"]["max_gpu_contact_pairs"]) == str:
            cfg["sim"]["physx"].update({"max_gpu_contact_pairs": int(eval(cfg["sim"]["physx"]["max_gpu_contact_pairs"]))})
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


# Math helpers


def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def set_sim_params_up_axis(sim_params: gymapi.SimParams, axis: str) -> int:
    """Set gravity based on up axis and return axis index.

    Args:
        sim_params: sim params to modify the axis for.
        axis: axis to set sim params for.
    Returns:
        axis index for up axis.
    """
    if axis == 'z':
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        return 2
    return 1


def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args, _ = parser.parse_known_args()

    args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args
