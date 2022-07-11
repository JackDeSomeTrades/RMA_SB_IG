from rma_sb_ig.utils.helpers import get_config, get_project_root, get_run_name, parse_config
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import zipfile

from box import Box
from rma_sb_ig.utils.dataloaders import RMAPhase2Dataset, RMAPhase2FastDataset
from rma_sb_ig.models.rma import RMAPhase2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Adaptation:
    def __init__(self, net, arch_config, tensorboard_log_writer=None):
        self.device = arch_config.device
        self.model = net(arch_config).to(self.device)
        self.model.double()
        self.epochs = arch_config.adaptation.epochs
        self.lr = arch_config.adaptation.lr
        if arch_config.adaptation.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
        if arch_config.adaptation.loss == 'mse':
            self.criterion = nn.MSELoss()

        if tensorboard_log_writer is not None:
            self.log_writer = tensorboard_log_writer

    def adapt(self, iterator):
        #n = iterator.
        n = len(iterator)
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0
            self.model.train()
            for label, data in tqdm(iterator, desc='|----', leave=False):
                data = data.squeeze()
                data = data.double()
                label = label.squeeze()
                label = label.double()
                # print(itr_cntr, data.shape, label.shape)

                data = torch.flatten(data, start_dim=2)

                zt = label.to(self.device)
                data = data.to(self.device)

                self.optimizer.zero_grad()

                zt_cap = self.model(data)
                loss = self.criterion(zt_cap, zt)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()/n
                if self.log_writer:
                    try:
                        self.log_writer.add_scalar('phase2/train/loss', epoch_loss, epoch)
                    except AttributeError:
                        self.log_writer.writer.add_scalar('phase2/train/loss', epoch_loss, epoch)
                        self.log_writer.writer.flush()

                # itr_cntr += 1
            # print(epoch_loss)

    def save(self, path:str, suffix='.zip'):
        """path should contain the path to the zip folder created by stable baselines. This function only adds the
        phase 2 adapted parameters into the same zip folder"""
        self.save_path = path+suffix
        with zipfile.ZipFile(self.save_path, mode='a') as archive:
            if self.model.state_dict() is not None:
                with archive.open('adaptation_module_parameters.pth', mode='w') as adapted_parameters:
                    torch.save(self.model.state_dict(), adapted_parameters)


if __name__ == '__main__':
    cfg = get_config('soto_task_rma_conf.yaml')
    hkl_fpath = '/home/student/Workspace/RMA_SB_IG/rma_sb_ig/output/PPO_4__soto.hkl'
    tb_logs = '/home/student/Workspace/RMA_SB_IG/rma_sb_ig/logs/PPO_4__soto_0'

    arch_config = Box(cfg).arch_config

    dataset_iterator = RMAPhase2FastDataset(hkl_filepath=hkl_fpath, device=arch_config.device,
                                            horizon=arch_config.state_action_horizon)

    # dataset_iterator = RMAPhase2Dataset(hkl_filepath=hkl_fpath, device=arch_config.device,
    #                                     horizon=arch_config.state_action_horizon)
    phase2dataloader = DataLoader(dataset_iterator, batch_size=128, pin_memory=False)
    logger = SummaryWriter(log_dir=tb_logs)

    model_adapted = Adaptation(net=RMAPhase2, arch_config=arch_config, tensorboard_log_writer=logger)
    model_adapted.adapt(phase2dataloader)





