import argparse
from utils.helpers import get_config,get_project_root
from models import rma
from box import Box

from utils.stable_baselines import RMAA1TaskVecEnvStableBaselineGym
from stable_baselines3 import PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='a1_task_rma')
    parser.add_argument('--logdir', '-l', type=str, default='outs/tb')

    args = parser.parse_args()
    cfg = get_config(f'{args.cfg}_conf.yaml')
    vec_env = RMAA1TaskVecEnvStableBaselineGym(cfg)

    # begin RL here
    rl_config = Box(cfg).rl_config    # convert config dict into namespaces
    arch = rma.Architecture(device=rl_config.device)
    policy_kwargs = arch.make_architecture()

    lr = eval(rl_config.ppo.lr)
    model = PPO(arch.policy_arch, vec_env, learning_rate=lr, verbose=1,
                tensorboard_log=rl_config.logging.dir.format(ROOT_DIR=get_project_root()), policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=rl_config.rl_timesteps, reset_num_timesteps=False)

    # model = PPO('MlpPolicy', env=vec_env, verbose=1, tensorboard_log=args.logdir, **cfg['rl']['ppo'])
    # model.learn(total_timesteps=cfg['rl']['total_timesteps'], callback=learn_cb, reset_num_timesteps=False)
