import os
import numpy as np

import argparse
from rma_sb_ig.utils.helpers import get_config, get_project_root, parse_config, UserNamespace, parse_replay_config, get_args
from box import Box
from rma_sb_ig.utils.stable_baselines import RMAA1TaskVecEnvStableBaselineGym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def get_updated_params_for_replay(args, cfg):
    env_cfg, sim_params, env_args = parse_replay_config(args, cfg)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    return env_cfg, sim_params, env_args


def play(args):
    cfg = get_config(f'{args.replay_cfg}_conf.yaml')
    env_cfg, sim_params, env_args = get_updated_params_for_replay(args, cfg)
    policy_save_fpath = os.path.join(os.getcwd(), f'experiments/{env_args.load_run}')

    # prepare environment
    replay_env = RMAA1TaskVecEnvStableBaselineGym((env_cfg, sim_params, env_args))
    obs = replay_env.get_observations()
    obs = obs.detach().cpu().numpy()    # for SB3 compatibility

    # load policy
    model = PPO.load(policy_save_fpath, env=replay_env, device=env_args.sim_device)

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    #
    for i in range(10 * int(replay_env.max_episode_length)):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = replay_env.step(action)
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(get_project_root(), 'output', env_args.load_run, 'exported',
                                        'frames', f"{img_idx}.png")
                replay_env.gym.write_viewer_image_to_file(replay_env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * replay_env.dt
            replay_env.set_camera(camera_position, camera_position + camera_direction)


if __name__ == '__main__':
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
