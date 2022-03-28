from rma_sb_ig.utils.helpers import get_config, get_project_root, get_run_name, parse_config
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from box import Box
from rma_sb_ig.utils.dataloaders import RMAPhase2Dataset
from rma_sb_ig.models.rma import RMAPhase2
from torch.utils.data import DataLoader


class Adaptation:
    def __init__(self, net, arch_config):
        self.device = arch_config.device
        self.model = net(arch_config).to(self.device)
        self.model.double()
        self.epochs = arch_config.adaptation.epochs
        self.lr = arch_config.adaptation.lr
        if arch_config.adaptation.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
        if arch_config.adaptation.loss == 'mse':
            self.criterion = nn.MSELoss()

    def adapt(self, iterator):
        for i in tqdm(range(self.epochs)):
            epoch_loss = 0
            epoch_acc = 0
            self.model.train()
            for data, label in iterator:
                data = data.squeeze()
                label = label.squeeze()

                zt = label.to(self.device)
                data = data.to(self.device)

                self.optimizer.zero_grad()

                zt_cap = self.model(data)
                loss = self.criterion(zt_cap, zt)

                # accuracy = self.calc_accuracy(zt_cap, zt)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                # epoch_acc += accuracy.item()


if __name__ == '__main__':
    cfg = get_config('a1_task_rma_conf.yaml')
    hkl_fpath = '/home/pavan/Workspace/RMA_SB_IG/rma_sb_ig/output/PPO_71.hkl'

    arch_config = Box(cfg).arch_config

    dataset_iterator = RMAPhase2Dataset(hkl_filepath=hkl_fpath, device=arch_config.device,
                                        horizon=arch_config.state_action_horizon)
    phase2dataloader = DataLoader(dataset_iterator)

    model_adapted = Adaptation(net=RMAPhase2, arch_config=arch_config)
    model_adapted.adapt(phase2dataloader)





