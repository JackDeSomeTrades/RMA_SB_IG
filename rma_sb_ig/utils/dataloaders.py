import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hickle as hkl
import torch.nn.functional as F


class RMAPhase2Dataset(Dataset):
    def __init__(self, hkl_filepath, device='cuda:0', horizon=50):
        self.device = device
        self.dataset = {}
        self.horizon = horizon
        full_hkl_data_dict = hkl.load(hkl_filepath)
        max_len_datadict = max(full_hkl_data_dict.keys())
        state_action_pair = torch.tensor((), device=device)
        actions = torch.tensor((), device=device)
        zt = torch.tensor((), device=device)

        # concat the state action pair and stack by step for the final tensor and
        # stack zt over steps
        for step in tqdm(range(1, max_len_datadict), desc='Reading state action data:'):
            state_at_step = full_hkl_data_dict[step]['state']
            action_at_step = full_hkl_data_dict[step]['actions']
            state_action_pair_at_step = torch.cat((state_at_step, action_at_step), dim=1)
            state_action_pair = torch.cat((state_action_pair, state_action_pair_at_step.unsqueeze(-1)), dim=-1)

            env_at_step = full_hkl_data_dict[step]['env_encoding']
            zt = torch.cat((zt, env_at_step.unsqueeze(-1)), dim=-1)

        self.dataset['zt'] = zt
        self.dataset['x_a'] = state_action_pair
        print("Ready")

    def __getitem__(self, idx):
        labels = self.dataset['zt'][:, :, idx]
        out = self.dataset['x_a'][:, :, idx:idx+self.horizon]
        pad_val = self.horizon - out.size()[-1]
        # print(pad_val)
        data = F.pad(out, (0, pad_val), mode='replicate')

        return labels, data

    def __len__(self):
        return self.dataset['zt'].size(dim=-1)


if __name__ == '__main__':
    hkl_fpath = '/home/pavan/Workspace/RMA_SB_IG/rma_sb_ig/output/PPO_71.hkl'
    dataset_main = RMAPhase2Dataset(hkl_filepath=hkl_fpath)

    phase2dataloader = DataLoader(dataset_main)

    for label, data in tqdm(phase2dataloader):
        label = label.squeeze()
        data = data.squeeze()
        pass



