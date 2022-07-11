import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hickle as hkl
import torch.nn.functional as F


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
        labels = self.dataset['zt'][:, :, idx].to(self.device)
        out = self.dataset['x_a'][:, :, idx:idx+self.horizon].to(self.device)
        pad_val = self.horizon - out.size()[-1]
        # print(pad_val)
        data = F.pad(out, (0, pad_val), mode='replicate')

        return labels, data

    def __len__(self):
        return self.dataset['zt'].size(dim=-1)


class RMAPhase2FastDataset(Dataset):
    def __init__(self, hkl_filepath, device='cuda', horizon=50):
        print(f"Reading dataset from:{hkl_filepath}, named: {hkl_filepath.split('/')[-1]}")
        self.device = device
        self.horizon = horizon
        self.full_hkl_data_dict = hkl.load(hkl_filepath)
        self._load_fast_dataset()
        self.state_action_pair = torch.zeros(self.fast_dataset.size()[0],self.fast_dataset.size()[1],self.horizon,device=self.device)
        print("Ready")

    def __getitem__(self, idx):

        bsup = self.horizon if idx+self.horizon <= self.num_element else self.horizon - (self.horizon + idx) % self.num_element
        if bsup == 50 :
            self.state_action_pair[:] = self.fast_dataset[...,idx:idx+self.horizon]
        else :
            self.state_action_pair[...,:bsup] = self.fast_dataset[...,idx:]
            self.state_action_pair[:,:, bsup:] = torch.unsqueeze(self.fast_dataset[:,:,-1],-1)

        # Check if there are 50 elements every time this is called. If there is a break in the previous loop, make sure\
        # the data is padded with the final state action pair.
        env_at_step = self.env[...,idx]
        label = env_at_step.unsqueeze(-1)   # zt

        return label, self.state_action_pair

    def __len__(self):
        return max(self.full_hkl_data_dict.keys())

    def _load_fast_dataset(self):
        self.fast_dataset = torch.tensor(())
        self.env = torch.tensor(())
        self.num_element = len(self.full_hkl_data_dict)
        print("sending dataset on GPU")
        for step in tqdm(range(1,len(self.full_hkl_data_dict)+1)):
            state_at_step = self.full_hkl_data_dict[step]['state']
            action_at_step = self.full_hkl_data_dict[step]['actions']
            env_at_step = self.full_hkl_data_dict[step]['env_encoding']
            state_action_pair_at_step = torch.cat((state_at_step, action_at_step), dim=1)
            self.fast_dataset = torch.cat((self.fast_dataset, state_action_pair_at_step.unsqueeze(-1)), dim=-1)
            self.env = torch.cat((self.env,env_at_step.unsqueeze(-1)),dim=-1)
        self.env = self.env.cuda(non_blocking = True)


if __name__ == '__main__':
    hkl_fpath = '/home/student/Workspace/RMA_SB_IG/rma_sb_ig/output/PPO_5__soto_0.hkl'
    # dataset_main = RMAPhase2Dataset(hkl_filepath=hkl_fpath)
    dataset_main = RMAPhase2FastDataset(hkl_filepath=hkl_fpath)

    phase2dataloader = DataLoader(dataset_main, batch_size=64)

    for label, data in tqdm(phase2dataloader):
        label = label.squeeze()
        data = data.squeeze()
        pass



