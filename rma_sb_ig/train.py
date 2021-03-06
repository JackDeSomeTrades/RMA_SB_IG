import argparse
import os
from box import Box

from rma_sb_ig.utils.helpers import get_config, get_project_root, get_run_name, parse_config
from rma_sb_ig.utils.trainers import Adaptation
from rma_sb_ig.utils.dataloaders import RMAPhase2Dataset, RMAPhase2FastDataset
from rma_sb_ig.models import rma
from rma_sb_ig.utils.stable_baselines import RMAA1TaskVecEnvStableBaselineGym, RMAV0TaskVecEnvStableBaselineGym, SaveHistoryCallback

from torch.utils.data import DataLoader
import torch
from stable_baselines3 import PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='a1_task_rma')
    parser.add_argument('--savedir', '-s', type=str, default='experiments/')
    parser.add_argument('--dsetsavedir', '-k', type=str, default='output/')
    parser.add_argument('--phase', '-p', type=str, default='2')
    parser.add_argument('--run_comment', '-m', type=str, default=None)
    parser.add_argument('--robot_name', '-r', type=str, default='a1')
    parser.add_argument('--timestamp', '-t', type=bool, default=False)

    args, _ = parser.parse_known_args()

    cfg = get_config(f'{args.cfg}_conf.yaml')
    robot_name = args.robot_name
    if robot_name == 'a1':
        vec_env = RMAA1TaskVecEnvStableBaselineGym(parse_config(cfg))
    elif robot_name == 'v0':
        vec_env = RMAV0TaskVecEnvStableBaselineGym(parse_config(cfg))

    # begin RL here
    # ----------- Configs -----------------#
    rl_config = Box(cfg).rl_config  # convert config dict into namespaces
    arch_config = Box(cfg).arch_config

    # ----------- Paths -------------------#

    run_name = get_run_name(rl_config, args)
    model_save_path = os.path.join(os.path.join(os.getcwd(), f'{args.savedir}'), run_name)
    intermediate_dset_save_path = os.path.join(os.getcwd(), f'{args.dsetsavedir}', run_name)+'.hkl'

    # ----------- Loaders and Callbacks ---#
    save_history_callback = SaveHistoryCallback(savepath=intermediate_dset_save_path)

    if args.phase == '1' or args.phase == None:
        # ----------------- RMA Phase 1 -------------------------------------------- #
        arch = rma.Architecture(arch_config=arch_config, device=arch_config.device)
        policy_kwargs = arch.make_architecture()

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

        model.learn(total_timesteps=rl_config.n_timesteps, reset_num_timesteps=False, tb_log_name=run_name, callback=save_history_callback)
        model.save(path=model_save_path)
        # need to close the sim env here to release GPU mem for the next phase.

        vec_env.close()
        torch.cuda.empty_cache()  # to see if everything is released after the first phase.

    if args.phase == '2' or args.phase == None:
        # ----------------- RMA Phase 2 -------------------------------------------- #
        dataset_iterator = RMAPhase2FastDataset(hkl_filepath=intermediate_dset_save_path, device=arch_config.device,
                                                horizon=arch_config.state_action_horizon)
        phase2dataloader = DataLoader(dataset_iterator)

        model_adapted = Adaptation(net=rma.RMAPhase2, arch_config=arch_config, tensorboard_log_writer=save_history_callback.tb_formatter)
        model_adapted.adapt(phase2dataloader)
        model_adapted.save(path=model_save_path)



