from torch import nn as nn

#%%

class Sampler:
    def __init__(self, agent:str, device:str):
        self.agent=agent
        self.device=device

    def sample(self):
        return dict(SAC=self.sample_sac_params, PPO=self.sample_ppo_params)[self.agent]()

    def sample_sac_params(self) -> (dict, dict):
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}["relu"]
        hyperparams_categorical = dict(gamma=[0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
                                    batch_size=[16, 32, 64, 128, 256, 512],
                                    buffer_size=[int(1e4), int(1e5), int(1e6)],
                                    learning_starts=[0, 100, 200, 300],
                                    train_freq=[1, 4, 8, 16, 32, 64, 128, 256, 512],
                                    tau=[0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

        hyperparams_loguniform = dict(learning_rate=[1e-6, 1e-3])
        hyperparams_uniform = dict(log_std_init=[-4, 1])
        hyperparam_range = dict(categorical=hyperparams_categorical,
                                loguniform=hyperparams_loguniform,
                                uniform=hyperparams_uniform)
        policy_kwargs = dict(net_arch=[400, 300], activation_fn=activation_fn)
        default_hyperparams = dict(policy='MlpPolicy',
                                   policy_kwargs=policy_kwargs,
                                   target_entropy="auto",
                                   ent_coef="auto",
                                   device=self.device)
        return default_hyperparams, hyperparam_range

    def sample_ppo_params(self) -> (dict, dict):
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}["relu"]
        hyperparams_categorical = dict(n_steps=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                                    gamma=[0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
                                    batch_size=[8, 16, 32, 64, 128, 256, 512],
                                    clip_range=[0.1, 0.2, 0.3, 0.4],
                                    n_epochs=[1, 5, 10, 20],
                                    gae_lambda=[0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
                                    max_grad_norm=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        hyperparams_loguniform = dict(learning_rate=[1e-6, 1e-3], ent_coef=[0.00000001, 0.1])
        hyperparams_uniform = dict(vf_coef=[0, 1])
        hyperparam_range = dict(categorical=hyperparams_categorical,
                                loguniform=hyperparams_loguniform,
                                uniform=hyperparams_uniform)
        policy_kwargs = dict(net_arch=[400, 300], activation_fn=activation_fn, ortho_init=False)
        default_hyperparams = dict(policy='MlpPolicy',
                                   policy_kwargs=policy_kwargs,
                                   device=self.device)
        return default_hyperparams, hyperparam_range




#%%

def sample_hyperparams(agent:str,  device:str="cuda:0")->(dict, dict):
    sampler=Sampler(agent=agent, device=device)
    default_hyperparams, hyperparam_range=sampler.sample()
    return default_hyperparams, hyperparam_range








