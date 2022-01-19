
from utils.stable_baselines import RMAA1TaskVecEnvStableBaselineGym
import argparse
from utils.helpers import get_config
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='a1_task_rma')
    parser.add_argument('--logdir', '-l', type=str, default='outs/tb')

    args = parser.parse_args()
    cfg = get_config(f'{args.cfg}_conf.yaml')

    vec_env = RMAA1TaskVecEnvStableBaselineGym(cfg)
    # print(vec_env)

    check_env(vec_env)


