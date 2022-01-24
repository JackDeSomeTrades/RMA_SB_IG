import torch
from torch import nn
# from configs import env_config
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import utils


class EnvironmentEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=40, save_dict=None, save_intermediate=False, device='cpu'):
        super(EnvironmentEncoder, self).__init__(observation_space, features_dim)
        self.device = device
        if save_intermediate:
            self.save_dict = save_dict
        else:
            if save_dict is not None:
                print("Set save_intermediate parameter to True")
                raise NotImplementedError

        self.zt = torch.empty((0, env_config.ENCODED_EXTRINSIC_SIZE), device=self.device)
        self.environmental_factor_encoder = nn.Sequential(
            nn.Linear(in_features=env_config.ENV_SIZE, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=env_config.ENCODED_EXTRINSIC_SIZE)
        )

    def forward(self, observations):
        et = observations[-env_config.ENV_SIZE:].device(self.device)
        self.zt = self.environment(et)

        # save zt for the second phase here.
        # depending on what saving mechanism is used here ( possibly hdf5)
        utils.update_encoding_storage(self.save_dict, self.zt)

        # concatenate the environmental encoding with the rest of the state variables.
        policy_input_variable = torch.cat((observations[0:env_config.ENV_SIZE,:], self.zt), -1)

        return policy_input_variable


class RMAPhase2(nn.Module):
    def __init__(self):
        super(RMAPhase2, self).__init__()
        self.adaptation_module = nn.Sequential(
            nn.Linear(in_features=(env_config.STATESPACE_SIZE + env_config.ACTIONSPACE_SIZE) * 50, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=32),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=4),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),

            nn.Flatten()
        )

        self.linear = nn.LazyLinear(out_features=env_config.ENCODED_EXTRINSIC_SIZE)  # automatically infers the input dim

    def forward(self, data):
        """in adaptation (phase2):
            data is a tuple of cat of 50 previous state action pair tensors and current statespace data as before
            data = (torch.cat((x_51, a_51),.....,(x_tprev, a_tprev),dim=0), statespace_t_prev)
        """

        temporal_data = data[0]
        intermediate = self.adaptation_module(temporal_data)
        z_cap_t = self.linear(intermediate)

        return z_cap_t


class Architecture():
    def __init__(self, device='cpu', savefile=None):
        self.device = device
        self.savefile = savefile

        self.features_encoder_arch = EnvironmentEncoder
        self.features_dim_encoder = env_config.ENCODED_EXTRINSIC_SIZE + env_config.ACTIONSPACE_SIZE + env_config.STATESPACE_SIZE

        self.policy_class = "MlpPolicy"
        self.policy_arch = [dict(pi=[256, 128], vf=[256, 128])]
        self.policy_activation_fn = nn.ReLU

    def make_architecture(self):
        return dict(
            features_extractor_class=EnvironmentEncoder,
            features_extractor_kwargs=dict(features_dim=self.features_dim_encoder, save_intermediate=True,
                                           device=self.device,save_dict=self.savefile),
            activation_fn=self.policy_activation_fn,
            net_arch=self.policy_arch)





