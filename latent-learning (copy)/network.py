import torch
import torch.nn as nn


'''
    Helper functions
'''
def create_mlp(layer_dims, act_fn = nn.ReLU):
    num_dims = len(layer_dims)
    assert num_dims >= 2

    layers = []
    for t in range(num_dims - 1):
        layers.append(nn.Linear(layer_dims[t], layer_dims[t + 1]))
        if t < num_dims - 2:
            layers.append(act_fn())

    return nn.Sequential(*layers)


'''
    Receive an input, then output a distribution by predicting the mean and log standard deviation.
'''
class MultivariateNormalDistribution(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims = [128, 128], mu_act_fn = None, fixed_std = None, clip_logstd : list = None) -> None:
        super().__init__()

        self.fixed_std = fixed_std
        self.mu_act_fn = mu_act_fn
        self.clip_logstd = clip_logstd

        layer_dims = [input_dim]
        layer_dims += hidden_dims
        layer_dims += [out_dim * 2] if fixed_std is None else [out_dim]
        self.main = create_mlp(layer_dims)


    def forward(self, x):
        x = self.main(x)
        
        if self.fixed_std is None:
            mu, logstd = torch.chunk(x, 2, -1)
            if self.clip_logstd is not None:
                assert len(self.clip_logstd) == 2
                logstd = torch.clip(logstd, *self.clip_logstd)
            std = torch.exp(logstd)
        else:
            mu = x
            std = self.fixed_std

        if self.mu_act_fn is not None:
            mu = self.mu_act_fn(mu)

        dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
        return dist, (mu, std)



class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims = [128, 128], out_act_fn = None) -> None:
        super().__init__()

        self.out_act_fn = out_act_fn

        layer_dims = [input_dim] + hidden_dims + [out_dim]
        self.main = create_mlp(layer_dims)


    def forward(self, x):
        x = self.main(x)

        if self.out_act_fn is not None:
            x = self.out_act_fn(x)

        return x


class BaseLatentModel(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dims = [128, 128]) -> None:
        super().__init__()

        self.o_encoder = MultivariateNormalDistribution(obs_dim, z_dim, hidden_dims)
        self.z_decoder = Decoder(z_dim, obs_dim, hidden_dims)


    def forward(self, o):
        z_dist, (z_mu, z_std) = self.o_encoder(o)
        z = z_dist.rsample()
        o_recon = self.z_decoder(z)
        return (z_dist, z_mu, z_std), o_recon




class TransitionLatentModel(nn.Module):
    def __init__(self, obs_dim, act_dim, z_dim, hidden_dims = [128, 128]) -> None:
        super().__init__()

        self.mlp_o = nn.Linear(obs_dim, hidden_dims[0] // 2)
        self.mlp_a = nn.Linear(act_dim, hidden_dims[0] // 2)
        self.z_encoder = MultivariateNormalDistribution(hidden_dims[0], z_dim, hidden_dims)
        self.z_decoder = MultivariateNormalDistribution(z_dim, obs_dim, hidden_dims)


    def forward(self, o, a):
        '''
            Input: o ~ [bs, obs_dim], a ~ [bs, act_dim]
            Output: latent z distribution, log probability of reconstructing o
        '''
        o_enc = self.mlp_o(o)
        a_enc = self.mlp_a(a)
        z_dist, (z_mu, z_std) = self.z_encoder(torch.cat([o_enc, a_enc], -1))
        z = z_dist.rsample()
        o_recon_dist, (o_recon_mu, o_recon_std) = self.z_decoder(z)
        return (z_dist, z_mu, z_std), (o_recon_dist, o_recon_mu, o_recon_std)



class SequentialLatentModel(nn.Module):
    def __init__(self, obs_dim, act_dim, z_dim, hidden_dims = [128, 128]) -> None:
        super().__init__()



if __name__ == '__main__':
    # dist = MultivariateNormalDistribution(8, 2)
    # x = torch.zeros((5, 8))
    # print(dist)
    # print(dist(x))

    model = TransitionLatentModel(8, 2, 4)
    o = torch.zeros((5, 8))
    a = torch.zeros((5, 2))
    print(model)
    print(model(o, a))