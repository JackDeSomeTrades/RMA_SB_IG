import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hickle as hkl
import torch.nn.functional as F
import pandas as pd

class RMAPhase2Dataset(Dataset):
    def __init__(self, hkl_filepath, device='cpu', horizon=50):
        """
        :param hkl_filepath: str
        :param device: cpu/cuda:0
        :param horizon: int
        Init the dataset in CPU memory and not GPU memory because we run out of GPU mem very quickly, especially if the
        total number of timesteps becomes too big. Convert to cuda while getting the item from the dataset.
        """
        self.device = device
        self.dataset = {}
        self.horizon = horizon
        full_hkl_data_dict = hkl.load(hkl_filepath)
        max_len_datadict = max(full_hkl_data_dict.keys())
        state_action_pair = torch.tensor(())
        actions = torch.tensor(())
        zt = torch.tensor(())

        # concat the state action pair and stack by step for the final tensor and
        # stack zt over steps
        df = pd.DataFrame(full_hkl_data_dict)
        df = df.T
        envs = torch.tensor(df['env_encoding'].values)
        states_actions =  df[['state','actions']].values
        for step in tqdm(range(1, max_len_datadict), desc='Reading state action data:'):
            state_at_step = full_hkl_data_dict[step]['state']
            action_at_step = full_hkl_data_dict[step]['actions']
            state_action_pair_at_step = torch.cat((state_at_step, action_at_step), dim=1)
            state_action_pair = torch.cat((state_action_pair, state_action_pair_at_step.unsqueeze(-1)), dim=-1)

            env_at_step = full_hkl_data_dict[step]['env_encoding']
            zt = torch.cat((zt, env_at_step.unsqueeze(-1)), dim=-1)

        self.dataset['zt'] = torch.cat(list(envs))
        self.dataset['x_a'] = torch.cat(list(states_actions))
        print("Ready")

    def __getitem__(self, idx):
        labels = self.dataset['zt'][:, :, idx].to(self.device)
        out = self.dataset['x_a'][:, :, idx:idx+self.horizon].to(self.device)
        pad_val = self.horizon - out.size()[-1]
        # print(pad_val)
        data = F.pad(out, (0, pad_val), mode='replicate')

        return labels, data

    def __len__(self):
        return self.dataset['zt'].size(dim=-1)


class RMAPhase2FastDataset(Dataset):
    def __init__(self, hkl_filepath, device='cpu', horizon=50):
        print(f"Reading dataset from:{hkl_filepath}, named: {hkl_filepath.split('/')[-1]}")
        self.device = device
        self.horizon = horizon
        self.full_hkl_data_dict = hkl.load(hkl_filepath)
        print("Ready")

    def __getitem__(self, idx):
        idx = idx+1   # quirk of saving the dataset, the step counter starts from 1, not 0
        state_action_pair = torch.tensor(()).cuda(non_blocking=True)
        for step in range(idx, idx+self.horizon):
            try:
                state_at_step = self.full_hkl_data_dict[step]['state'].cuda(non_blocking=True)
                action_at_step = self.full_hkl_data_dict[step]['actions'].cuda(non_blocking=True)
                state_action_pair_at_step = torch.cat((state_at_step, action_at_step), dim=1).cuda(non_blocking=True)
            except KeyError:
                # on the first key error (i.e., when there are less than 50 elements left in the dict), break the loop
                break
            state_action_pair = torch.cat((state_action_pair, state_action_pair_at_step.unsqueeze(-1)), dim=-1).cuda(non_blocking=True)
        # Check if there are 50 elements every time this is called. If there is a break in the previous loop, make sure\
        # the data is padded with the final state action pair.
        pad_val = self.horizon - state_action_pair.size()[-1]
        data = F.pad(state_action_pair, (0, pad_val), mode='replicate')
        env_at_step = self.full_hkl_data_dict[idx]['env_encoding'].cuda(non_blocking=True)
        label = env_at_step.unsqueeze(-1)   # zt

        return label, data

    def __len__(self):
        return max(self.full_hkl_data_dict.keys())

if __name__ == '__main__':
    hkl_fpath = '/home/stone/Workspace/RMA_SB_IG/rma_sb_ig/output/PPO_6__soto_0.hkl'
    # dataset_main = RMAPhase2Dataset(hkl_filepath=hkl_fpath)
    dataset_main = RMAPhase2FastDataset(hkl_filepath=hkl_fpath)

    phase2dataloader = DataLoader(dataset_main, batch_size=64)

    for label, data in tqdm(phase2dataloader):
        label = label.squeeze()
        data = data.squeeze()
        pass