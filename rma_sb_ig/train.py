import argparse
import os
from rma_sb_ig.utils.helpers import get_config, get_project_root, get_run_name, parse_config
from rma_sb_ig.models import rma
from box import Box

from rma_sb_ig.utils.stable_baselines import RMAA1TaskVecEnvStableBaselineGym, SaveHistoryCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='a1_task_rma')
    parser.add_argument('--savedir', '-s', type=str, default='experiments/')

    args = parser.parse_args()
    cfg = get_config(f'{args.cfg}_conf.yaml')

    vec_env = RMAA1TaskVecEnvStableBaselineGym(parse_config(cfg))
    # eval_env = RMAA1TaskVecEnvStableBaselineGym(cfg)  # for evaluating the performance of learning with SB3, not for learning.

    # begin RL here
    rl_config = Box(cfg).rl_config    # convert config dict into namespaces
    arch_config = Box(cfg).arch_config
    arch = rma.Architecture(arch_config=arch_config, device=rl_config.device)
    policy_kwargs = arch.make_architecture()

    run_name = get_run_name(rl_config)
    model_save_path = os.path.join(os.path.join(os.getcwd(), f'{args.savedir}'), run_name)

    # evaluation of learning performance here.
    # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=5000, deterministic=True, render=False)

    save_history_callback = SaveHistoryCallback()

    model = PPO(arch.policy_class, vec_env,
                learning_rate=eval(rl_config.ppo.lr), n_steps=rl_config.ppo.n_steps,
                batch_size=rl_config.ppo.batch_size, n_epochs=rl_config.ppo.n_epochs,
                gae_lambda=rl_config.ppo.gae_lambda, gamma=rl_config.ppo.gamma,
                clip_range=rl_config.ppo.clip_range, max_grad_norm=rl_config.ppo.max_grad_norm,
                ent_coef=rl_config.ppo.ent_coef, vf_coef=rl_config.ppo.vf_coef,
                # use_sde=rl_config.ppo.use_sde, sde_sample_freq=rl_config.ppo.sde_sample_freq,
                verbose=1,
                tensorboard_log=rl_config.logging.dir.format(ROOT_DIR=get_project_root()),
                policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=rl_config.n_timesteps, reset_num_timesteps=False, tb_log_name=run_name, callback=save_history_callback)  #, callback=eval_callback
    model.save(path=model_save_path)

