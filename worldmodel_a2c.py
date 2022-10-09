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
            nn.Linear(hidden_dim, obs_dim)
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
            nn.Linear(hidden_dim, 1)
        )


    def actor_critic(self, z_t, deterministic = False):
        a_mu = self.actor(z_t)
        a_dist = torch.distributions.Normal(a_mu, self.fixed_action_std)
        a = a_mu if deterministic else a_dist.sample()   
        a_logprob = a_dist.log_prob(a).sum(-1)
        v = self.critic(z_t).squeeze(-1)
        return a_dist, a, a_logprob, v


    def step(self, o_curr, a_prev, h_prev, s_prev, deterministic = False):
        posterior, (h_curr, s_curr) = self.rssm.posterior_forward(s_prev, a_prev, o_curr, h_prev)
        z_curr = torch.cat([h_curr, s_curr], -1)
        _, a, a_logprob, v = self.actor_critic(z_curr, deterministic)
        return posterior, a, a_logprob, v, (h_curr, s_curr)


    def forward(self, batch_obs_curr, batch_act_prev):
        bs = batch_obs_curr.size(1)
        h_t, s_t = self.rssm.init_states(bs)
        features, priors, posteriors = self.rssm(batch_obs_curr, batch_act_prev, h_t, s_t)
        recon_obs_curr = self.obs_recon_net(features)
        batch_a_dist, _, _, batch_v_pred = self.actor_critic(features)
        batch_a_logprob = batch_a_dist.log_prob(batch_act_prev).sum(-1)
        return features, priors, posteriors, recon_obs_curr, batch_a_dist, batch_a_logprob, batch_v_pred


    def get_parameter_dict(self):
        return {
            'actor' : self.actor.parameters(),
            'critic' : self.critic.parameters(),
            'worldmodel' : list(self.obs_recon_net.parameters()) + list(self.rssm.parameters())
        }



def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha = 1, lw = 1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha = 1, lw = 1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw = 2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left = 0, right = len(ave_grads))
    plt.ylim(bottom = -0.001, top = 0.002) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw = 4),
                Line2D([0], [0], color="b", lw = 4),
                Line2D([0], [0], color="k", lw = 4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig('GradientFlow.png')



if __name__ == '__main__':
    # s = GRUStack(input_dim = 4, hidden_dim = 10, num_layers = 2)
    # x = torch.zeros((3, 4))
    # h = torch.zeros((3, 10))
    # print(s(x, h).size())

    # net = RSSM(obs_dim = 4, act_dim = 2, state_dim = 8, hidden_dim = 8, num_gru_layers = 2)
    # h, s = net.init_states(3)
    # batch_obs = torch.zeros((5, 3, 4))
    # batch_act = torch.zeros((5, 3, 2))
    # features, priors, posteriors = net(batch_obs, batch_act, h, s)
    # print(features.size())
    # print(priors.size())
    # print(posteriors.size())

    # model = WorldModel(obs_dim = 4, act_dim = 2, state_dim = 8, hidden_dim = 8, num_gru_layers = 2)
    # h = torch.zeros((3, 8))
    # s = torch.zeros((3, 8))
    # batch_obs = torch.zeros((5, 3, 4))
    # batch_act = torch.zeros((5, 3, 2))
    # loss, loss_dict = model(batch_obs, batch_act, h, s)
    # print(loss)
    # print(loss_dict)

    model = WorldModelA2C(
        obs_dim = 4, 
        act_dim = 2, 
        state_dim = 8, 
        hidden_dim = 8, 
        num_gru_layers = 2,
        min_state_std = 0.1, 
        max_state_std = 1,
        fixed_action_std = 0.5
    )
    actor_critic_params = model.get_parameter_dict()
    optimizer = torch.optim.Adam([
        {'params' : actor_critic_params['actor'], 'lr' : 0.0003},
        {'params' : actor_critic_params['critic'], 'lr' : 0.001},
        {'params' : actor_critic_params['worldmodel'], 'lr' : 0.0001},
    ])

    # # Test
    # o_curr = torch.zeros((3, 4))
    # a_prev = torch.zeros((3, 2))
    # h_prev = torch.zeros((3, 8))
    # s_prev = torch.zeros((3, 8))
    # posterior, a, a_logprob, v, (h_curr, s_curr) = model.step(o_curr, a_prev, h_prev, s_prev, deterministic = False)

    # batch_o_curr = torch.zeros((5, 3, 4))
    # batch_a_prev = torch.zeros((5, 3, 2))
    # features, priors, posteriors, recon_obs_curr, batch_a_dist, batch_a_logprob, batch_v_pred = model(batch_o_curr, batch_a_prev)

    # z_t = torch.cat([h_prev, s_prev], -1)
    # a_dist, a, a_logprob, v = model.actor_critic(z_t)


    # Visualize gradient flow
    batch_obs = torch.randn((10, 3, 4))
    batch_next_obs = torch.randn((10, 3, 4))
    batch_actions = torch.randn((10, 3, 2))
    batch_action_logprobs = torch.randn((10, 3))
    batch_returns = torch.randn((10, 3))
    batch_advs = torch.randn((10, 3))
    batch_traj_masks = torch.ones((10, 3))
    _, priors, posteriors, recon_next_obs, batch_new_a_dist, batch_new_a_logprob, batch_v_pred = model(batch_next_obs, batch_actions)

    # Clip loss
    norm_advs = (batch_advs - batch_advs.mean()) / (batch_advs.std() + 1e-8)
    ratio = (batch_new_a_logprob - batch_action_logprobs).exp()
    surr1 = ratio * norm_advs
    surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * norm_advs
    clip_loss = -(torch.min(surr1, surr2) * batch_traj_masks).mean()

    # Entropy loss
    entropy_loss = -batch_new_a_dist.entropy().mean()

    # KL divergence loss
    priors_dist = model.rssm.create_diag_normal(priors)
    posteriors_dist = model.rssm.create_diag_normal(posteriors)
    kl_div_loss = torch.distributions.kl.kl_divergence(priors_dist, posteriors_dist).sum(0).mean()

    # Reconstruction observation loss
    recon_loss = 0.5 * (recon_next_obs - batch_next_obs).pow(2).sum([0, 2]).mean()

    # Critic loss
    critic_loss = 0.5 * ((batch_returns - batch_v_pred) * batch_traj_masks).pow(2).mean()

    # Optimize
    optimizer.zero_grad()
    total_loss = clip_loss + entropy_loss + kl_div_loss + recon_loss + critic_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.05)
    plot_grad_flow(model.named_parameters())
    optimizer.step()