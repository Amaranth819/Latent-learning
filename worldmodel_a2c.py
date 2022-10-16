from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class GRUStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 2) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        gru_layer_size = hidden_dim // num_layers

        self.gru = []
        self.gru.append(nn.GRUCell(input_dim, gru_layer_size))
        for _ in range(num_layers - 1):
            self.gru.append(nn.GRUCell(gru_layer_size, gru_layer_size))
        self.gru = nn.ModuleList(self.gru)


    def forward(self, x, h_prev):
        '''
            Input: 
                x: (bs, N)
                h_prev: (bs, hidden_dim)
            Output:
                h: (bs, N)
        '''
        h_prev_layers = torch.chunk(h_prev, self.num_layers, -1)
        h_curr = []
        for i in range(self.num_layers):
            x = self.gru[i](x, h_prev_layers[i])
            h_curr.append(x)
        return torch.cat(h_curr, -1)
        

    


class RSSM(nn.Module):
    def __init__(self, 
            obs_dim, 
            act_dim, 
            state_dim, 
            hidden_dim = 128, 
            num_gru_layers = 2,
            min_state_std = 0.1,
            max_state_std = 0.2
        ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.min_state_std = min_state_std
        self.max_state_std = max_state_std

        # Encoders
        self.mlp_s = nn.Linear(state_dim, hidden_dim // 2)
        self.mlp_a = nn.Linear(act_dim, hidden_dim // 2)

        # Deterministic state model
        self.gru = GRUStack(
            input_dim = hidden_dim,
            hidden_dim = state_dim,
            num_layers = num_gru_layers
        )

        # Stochastic state model: posterior
        self.posterior_mlp_o = nn.Linear(obs_dim, hidden_dim // 2)
        self.posterior_mlp_h = nn.Linear(state_dim, hidden_dim // 2)
        self.posterior_net = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps = 1e-3),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim * 2)
        )

        # Stochastic state model: prior
        self.prior_mlp_h = nn.Linear(state_dim, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps = 1e-3),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim * 2)
        )


    def create_diag_normal(self, p):
        mu, std = torch.chunk(p, 2, -1)
        std = self.min_state_std + (self.max_state_std - self.min_state_std) * torch.sigmoid(std)
        # dist = torch.distributions.MultivariateNormal(mu, torch.diag_embed(std))
        dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
        return dist


    def init_states(self, batch_size):
        '''
            Return: (h, s)
        '''
        device = next(self.gru.parameters()).device
        return (
            torch.zeros((batch_size, self.state_dim)).to(device),
            torch.zeros((batch_size, self.state_dim)).to(device),
        )


    def posterior_forward(self, s_prev, a_prev, o_curr, h_prev):
        '''
            Input:
                s_prev: (bs, s_dim)
                a_prev: (bs, act_dim)
                o_curr: (bs, obs_dim)
                h_prev: (bs, state_dim)
            Return:
                posterior: (bs, 2 * state_dim)
                h_curr: h_t = f(h_t-1, s_t-1, a_t-1)
                s_curr = q(s_t|h_t, o_t)
        '''
        x = torch.cat([self.mlp_s(s_prev), self.mlp_a(a_prev)], -1)
        h_curr = self.gru(x, h_prev)
        p = torch.cat([self.posterior_mlp_h(h_curr), self.posterior_mlp_o(o_curr)], -1)
        posterior = self.posterior_net(p)
        posterior_dist = self.create_diag_normal(posterior)
        s_curr = posterior_dist.rsample()
        return posterior, (h_curr, s_curr)


    def prior_forward(self, s_prev, a_prev, h_prev):
        '''
            Input:
                s_prev: (bs, s_dim)
                a_prev: (bs, act_dim)
                h_prev: (bs, state_dim)
            Return:
                prior: (bs, 2 * state_dim)
                h_curr: h_t = f(h_t-1, s_t-1, a_t-1)
                s_curr: q(s_t|h_t)
        '''
        x = torch.cat([self.mlp_s(s_prev), self.mlp_a(a_prev)], -1)
        h_curr = self.gru(x, h_prev)
        p = self.prior_mlp_h(h_curr)
        prior = self.prior_net(p)
        prior_dist = self.create_diag_normal(prior)
        s_curr = prior_dist.rsample()
        return prior, (h_curr, s_curr)


    def sequential_prior_forward(self, batch_h):
        '''
            batch_h: (seq_len, bs, state_dim)
        '''
        seq_len = batch_h.size(0)
        hs = batch_h.view(-1, self.state_dim)
        hs = self.prior_mlp_h(hs)
        batch_prior = self.prior_net(hs).view(seq_len, -1, self.state_dim * 2)
        return batch_prior


    def forward(self, batch_obs_curr, batch_act_prev, h_init, s_init):
        '''
            Inputs:
                batch_next_obs: (seq_len, bs, obs_dim)
                batch_act: (seq_len, bs, act_dim)
        '''
        posteriors = []
        hs = []
        ss = []
        h_t, s_t = h_init, s_init

        for t in range(batch_obs_curr.size(0)):
            posterior, (h_t, s_t) = self.posterior_forward(s_t, batch_act_prev[t], batch_obs_curr[t], h_t) # posterior at time t
            posteriors.append(posterior)
            hs.append(h_t)
            ss.append(s_t)

        hs = torch.stack(hs)
        ss = torch.stack(ss)
        features = torch.cat([hs, ss], -1)
        priors = self.sequential_prior_forward(hs) # prior at time t+1, minimize the KL divergence between prior_t+1 and posterior_t
        posteriors = torch.stack(posteriors)

        return features, priors, posteriors


    



class WorldModel(nn.Module):
    def __init__(
            self,
            obs_dim, 
            act_dim, 
            state_dim = 128, 
            hidden_dim = 128, 
            num_gru_layers = 2,
            min_state_std = 0.1,
            max_state_std = 1.0) -> None:
        super().__init__()

        self.rssm = RSSM(
            obs_dim, 
            act_dim, 
            state_dim, 
            hidden_dim, 
            num_gru_layers,
            min_state_std,
            max_state_std
        )

        self.obs_recon_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
    #     self.init_weight()
        
        
    # def init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # For FC
    #             torch.nn.init.normal_(m.weight.data, mean = 0, std = 1)
    #             if m.bias is not None:
    #                 torch.nn.init.zeros_(m.bias.data)
    #         elif isinstance(m, (nn.LSTM, nn.GRU)):
    #             # For recurrent network
    #             for name, param in m.named_parameters():
    #                 if 'weight' in name:
    #                     torch.nn.init.orthogonal_(param)
    #                 elif 'bias' in name:
    #                     torch.nn.init.zeros_(param)


    def forward(self, batch_obs_curr, batch_act_prev, h_init, s_init):
        features, priors, posteriors = self.rssm(batch_obs_curr, batch_act_prev, h_init, s_init)
        recon_obs_curr = self.obs_recon_net(features)

        # Compute loss
        recon_obs_loss = (recon_obs_curr - batch_obs_curr).pow(2).sum([0, 2])
        
        priors_dist = self.rssm.create_diag_normal(priors)
        posteriors_dist = self.rssm.create_diag_normal(posteriors)
        kl_div_loss = torch.distributions.kl.kl_divergence(priors_dist, posteriors_dist).sum(0)

        total_loss = (recon_obs_loss + kl_div_loss).mean()

        return total_loss, {'recon_obs_loss' : recon_obs_loss.mean().item(), 'kl_div_loss' : kl_div_loss.mean().item(), 'total_loss' : total_loss.item()}




class WorldModelA2C(nn.Module):
    def __init__(
            self,
            obs_dim, 
            act_dim, 
            state_dim = 128, 
            hidden_dim = 128, 
            num_gru_layers = 2,
            min_state_std = 0.1,
            max_state_std = 1.0,
            fixed_action_std = 0.5) -> None:
        super().__init__()

        self.rssm = RSSM(obs_dim, act_dim, state_dim, hidden_dim, num_gru_layers, min_state_std, max_state_std)

        self.obs_recon_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim * 2)
        )

        self.actor = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )
        self.fixed_action_std = fixed_action_std

        self.critic = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )


    # def actor_critic(self, z_t, deterministic = False):
    #     a_mu = self.actor(z_t)
    #     # a_dist = torch.distributions.Normal(a_mu, self.fixed_action_std)
    #     a_dist = torch.distributions.Independent(torch.distributions.Normal(a_mu, self.fixed_action_std), 1)
    #     a = a_mu if deterministic else a_dist.sample()   
    #     a_logprob = a_dist.log_prob(a)
    #     v = self.critic(z_t).squeeze(-1)
    #     return a_dist, a, a_logprob, v


    def actor_forward(self, z_t, a = None, deterministic = False):
        a_mu = self.actor(z_t)
        a_dist = torch.distributions.Independent(torch.distributions.Normal(a_mu, self.fixed_action_std), 1)
        if a is not None:
            a_logprob = a_dist.log_prob(a)
            return None, a_logprob, a_dist.entropy()
        else:
            a = a_mu if deterministic else a_dist.sample()   
            a_logprob = a_dist.log_prob(a)
            return a, a_logprob, a_dist.entropy()


    def critic_forward(self, z_t, v = None):
        x = self.critic(z_t)
        mu, std = torch.chunk(x, 2, -1) 
        std = torch.sigmoid(std) + 0.01
        v_dist = torch.distributions.Independent(torch.distributions.Normal(mu.squeeze(-1), std.squeeze(-1)), 1)
        if v is not None:
            v_logprob = v_dist.log_prob(v)
            return None, v_logprob
        else:
            v = v_dist.sample()
            v_logprob = v_dist.log_prob(v)
            return v, v_logprob


    def obs_recon_forward(self, z_t, obs):
        x = self.obs_recon_net(z_t)
        mu, std = torch.chunk(x, 2, -1)
        std = torch.sigmoid(std) + 0.01
        obs_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
        obs_logprob = obs_dist.log_prob(obs)
        return obs_logprob


    def step(self, o_curr, a_prev, h_prev, s_prev, deterministic = False):
        posterior, (h_curr, s_curr) = self.rssm.posterior_forward(s_prev, a_prev, o_curr, h_prev)
        z_curr = torch.cat([h_curr, s_curr], -1)
        a, a_logprob, _ = self.actor_forward(z_curr, deterministic = deterministic)
        v, _ = self.critic_forward(z_curr)
        return posterior, a, a_logprob, v, (h_curr, s_curr)


    def forward(self, batch_obs_curr, batch_act_prev, batch_returns):
        bs = batch_obs_curr.size(1)
        h_t, s_t = self.rssm.init_states(bs)
        features, priors, posteriors = self.rssm(batch_obs_curr, batch_act_prev, h_t, s_t)
        batch_recon_obs_logprob = self.obs_recon_forward(features, batch_obs_curr)
        _, batch_a_logprob, batch_a_entropy = self.actor_forward(features, batch_act_prev)
        _, batch_v_logprob = self.critic_forward(features, batch_returns)

        return features, priors, posteriors, batch_recon_obs_logprob, batch_a_logprob, batch_a_entropy, batch_v_logprob


    def get_parameter_dict(self):
        return {
            'actor' : self.actor.parameters(),
            'critic' : self.critic.parameters(),
            'worldmodel' : list(self.obs_recon_net.parameters()) + list(self.rssm.parameters())
        }




'''
    10.13
'''
class dSSMVAE(nn.Module):
    def __init__(self, act_dim, state_dim = 128, hidden_dim = 128) -> None:
        super().__init__()

        self.mlp_a = nn.Linear(act_dim, hidden_dim)
        self.gru = GRUStack(hidden_dim, state_dim)
        self.latent_z = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)
        )


    def step(self, s_prev, a_prev):
        '''
            z_t ~ p(z_t|s_t-1, a_t-1)
            s_t = f(s_t-1, a_t-1)
        '''
        enc_a_prev = self.mlp_a(a_prev)
        z_curr_p = self.latent_z(torch.cat([s_prev, enc_a_prev], -1))
        z_curr_mu, z_curr_std = torch.chunk(z_curr_p, 2, -1)
        z_curr_std = torch.sigmoid(z_curr_std) + 0.01
        z_curr_dist = torch.distributions.Independent(torch.distributions.Normal(z_curr_mu, z_curr_std), 1)
        z_curr = z_curr_dist.sample()
        s_curr = self.gru(enc_a_prev, s_prev)
        return z_curr_dist, (z_curr, s_curr)



class sSSMVAE(nn.Module):
    def __init__(self, act_dim, state_dim = 128, hidden_dim = 128) -> None:
        super().__init__()

        self.mlp_a = nn.Linear(act_dim, hidden_dim)
        self.mlp_z = nn.Linear(state_dim, hidden_dim)
        self.gru = GRUStack(hidden_dim * 2, state_dim)
        self.latent_z = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)
        )

    
    def step(self, s_prev, a_prev):
        '''
            z_t ~ p(z_t|s_t-1, a_t-1)
            s_t = f(s_t-1, a_t-1, z_t)
        '''
        enc_a_prev = self.mlp_a(a_prev)
        z_curr_p = self.latent_z(torch.cat([s_prev, enc_a_prev], -1))
        z_curr_mu, z_curr_std = torch.chunk(z_curr_p, 2, -1)
        z_curr_std = torch.sigmoid(z_curr_std) + 0.01
        z_curr_dist = torch.distributions.Independent(torch.distributions.Normal(z_curr_mu, z_curr_std), 1)
        z_curr = z_curr_dist.sample()

        enc_z_curr = self.mlp_z(z_curr)
        s_curr = self.gru(torch.cat([enc_a_prev, enc_z_curr], -1), s_prev)
        return z_curr_dist, (z_curr, s_curr)



if __name__ == '__main__':
    dssm = dSSMVAE(2, 8, 8)
    sssm = sSSMVAE(2, 8, 8)
    s = torch.zeros((3, 8))
    a = torch.zeros((3, 2))
    dssm.step(s, a)
    sssm.step(s, a)