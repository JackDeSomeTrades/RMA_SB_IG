import argparse
from utils.helpers import get_config

from utils.stable_baselines import RMAA1TaskVecEnvStableBaselineGym

# from stable_baselines3 import PPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='a1_task_rma')
    parser.add_argument('--logdir', '-l', type=str, default='outs/tb')
    # parser.add_argument()
    # parser.add_argument()
    args = parser.parse_args()

    cfg = get_config(f'{args.cfg}_conf.yaml')

    vec_env = RMAA1TaskVecEnvStableBaselineGym(cfg)


    model = PPO('MlpPolicy', env=vec_env, verbose=1, tensorboard_log=args.logdir, **cfg['rl']['ppo'])
    # model.learn(total_timesteps=cfg['rl']['total_timesteps'], callback=learn_cb, reset_num_timesteps=False)