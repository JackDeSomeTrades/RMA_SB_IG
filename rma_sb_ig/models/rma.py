from rma_sb_ig.utils.helpers import get_config
import torch
from torch import nn
# from configs import env_config
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchinfo import summary
from box import Box


class EnvironmentEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, arch_config, features_dim=40, save_dict=None, save_intermediate=False, device='cpu'):
        super(EnvironmentEncoder, self).__init__(observation_space, features_dim)
        self.device = device
        self.arch_config = arch_config
        if save_intermediate:
            self.save_dict = save_dict
        else:
            if save_dict is not None:
                print("Set save_intermediate parameter to True")
                raise NotImplementedError

        self.zt = torch.empty((0, self.arch_config.encoder.encoded_extrinsic_size), device=self.device)
        self.environmental_factor_encoder = nn.Sequential(
            nn.Linear(in_features=self.arch_config.encoder.env_size, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=self.arch_config.encoder.encoded_extrinsic_size)
        )

    def forward(self, observations):
        et = observations[:, -self.arch_config.encoder.env_size:]
        self.zt = self.environmental_factor_encoder(et)

        # concatenate the environmental encoding with the rest of the state variables.
        policy_input_variable = torch.cat((observations[:, :-self.arch_config.encoder.env_size], self.zt), -1)

        return policy_input_variable


class RMAPhase2(nn.Module):
    def __init__(self, arch_config):
        super(RMAPhase2, self).__init__()
        self.adaptation_module_lin = nn.Sequential(
            # nn.Linear(in_features=(arch_config.encoder.state_space_size + arch_config.encoder.action_space_size) * arch_config.state_action_horizon, out_features=128),
            nn.LazyLinear(out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=32)
        )
        self.adaptation_module_conv = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=4),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.Flatten()
        )

        self.linear = nn.LazyLinear(out_features=arch_config.encoder.encoded_extrinsic_size)  # automatically infers the input dim

    def forward(self, data):
        """in adaptation (phase2):
            data contains past timesteps of the robot state-action pair. If the shape of the data is (1024 x 42 x 50):
            1024 is the number of environments derived from isaac gym, 42 is the size of the ( state + action) space and
            50 is the time horizon of state evolution.
        """
        intermediate = self.adaptation_module_lin(data)
        intermediate = intermediate.unsqueeze(dim=3)
        intermediate = intermediate.view(intermediate.size(0)*intermediate.size(1), intermediate.size(2), 1)
        conv_intermediate = self.adaptation_module_conv(intermediate)
        z_cap_t = self.linear(conv_intermediate)

        z_cap_t = z_cap_t.view(data.size(0), -1, z_cap_t.size(-1))
        return z_cap_t


class Phase2Net(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv1d(1, 32, 3, stride=3)
        conv2 = nn.Conv1d(32, 64, 5, stride=3)
        conv3 = nn.Conv1d(64, 128, 5, stride=2)

        conv4 = nn.Conv1d(128, 64, 7, stride=2)
        conv5 = nn.Conv1d(64, 64, 5, stride=2)
        conv6 = nn.Conv1d(64, 32, 5, stride=1)
        conv7 = nn.Conv1d(32, 32, 3, stride=1)

        self.layers = nn.Sequential(conv1, nn.BatchNorm1d(32), nn.ReLU(),
                                    conv2, nn.BatchNorm1d(64), nn.ReLU(),
                                    conv3, nn.BatchNorm1d(128), nn.ReLU(),
                                    conv4, nn.BatchNorm1d(64), nn.ReLU(),
                                    conv5, nn.BatchNorm1d(64), nn.ReLU(),
                                    conv6, nn.BatchNorm1d(32), nn.ReLU(),
                                    conv7, nn.Flatten())

    def forward(self, x):
        x = self.layers(x)
        return x


class Architecture():
    def __init__(self, arch_config, device='cpu', savefile=None, encoder=True):
        self.device = device
        self.savefile = savefile
        self.arch_config = arch_config
        self.encoder = encoder
        if self.encoder :
            self.features_encoder_arch = EnvironmentEncoder
            self.features_dim_encoder = self.arch_config.encoder.encoded_extrinsic_size + self.arch_config.encoder.action_space_size + self.arch_config.encoder.state_space_size

        self.policy_class = "MlpPolicy"
        self.policy_arch = [dict(pi=[400, 300], vf=[400, 300])]
        self.policy_activation_fn = nn.ELU

    def make_architecture(self):
        if self.encoder :
            return dict(
                features_extractor_class=EnvironmentEncoder,
                features_extractor_kwargs=dict(features_dim=self.features_dim_encoder, save_intermediate=True,
                                            device=self.device, save_dict=self.savefile, arch_config=self.arch_config),
                activation_fn=self.policy_activation_fn,
                net_arch=self.policy_arch)
        else : 
            return dict(
                activation_fn=self.policy_activation_fn,
                net_arch=self.policy_arch)


def test_net():
    cfg = 'soto_conf.yaml'
    cfg = get_config(cfg)
    arch_cfg = Box(cfg).arch_config
    net = RMAPhase2(arch_config=arch_cfg)
    input_data_size = (64, 1024, 2100)
    summary(net, input_size=input_data_size)


if __name__ == '__main__':
    test_net()
