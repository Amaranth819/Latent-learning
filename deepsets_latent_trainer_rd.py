import torch
import torch.nn as nn
import gym
import numpy as np
import os
import itertools
from network import TransitionLatentModel, BaseLatentModel
from base_trainer import BaseTrainer
from random_envs.halfcheetah import HalfCheetahEnv_RandomDynamics
from random_envs.dynamics_util import change_dynamics, scale_dynamics, get_dynamics


class DeepsetsLatentTrainer(BaseTrainer):
    def __init__(self, env_params, model_params, optim_params, training_params, device='auto') -> None:
        super().__init__(env_params, model_params, optim_params, training_params, device)

        joint_damping_ratios = (0.98, 1.0)
        body_mass_ratios = (0.98, 1.0)

        scaled_joint_damping = [scale_dynamics(self.env.vec_envs[0].original_joint_damping, r) for r in joint_damping_ratios]
        scaled_body_mass = [scale_dynamics(self.env.vec_envs[0].original_body_mass, r) for r in body_mass_ratios]

        for i, (sjd, sbm) in enumerate(itertools.product(scaled_joint_damping, scaled_body_mass)):
            self.env.vec_envs[i].set_dynamics('joint', 'damping', sjd)
            self.env.vec_envs[i].set_dynamics('body', 'mass', sbm)



    def create_model(self, model_params, optim_params):
        hidden_size = 256
        p_dim = 64
        self.domain_learner = nn.Sequential(
            nn.Linear(self.env.obs_dim * 2 + self.env.act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, p_dim)
        ).to(self.device)

        self.reconstructor = nn.Sequential(
            nn.Linear(self.env.obs_dim + self.env.act_dim + p_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.env.obs_dim)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            [{'params' : self.domain_learner.parameters(), 'lr' : 0.0001},
            {'params' : self.reconstructor.parameters(), 'lr' : 0.0005}], 
            eps = 1e-5
        )


    def get_cpu_named_parameters(self):
        return list(self.domain_learner.cpu().named_parameters()) + list(self.reconstructor.cpu().named_parameters())
        # return list(self.reconstructor.cpu().named_parameters())


    def predict(self, obs_ts, deterministic = False):
        pass


    def train(self, batch):
        obs, next_obs, action, _, _ = batch
        T = obs.size(0)

        ps = [self.domain_learner(torch.cat([obs[t], next_obs[t], action[t]], -1)) for t in range(T)]
        p = torch.stack(ps).mean(0, keepdim = True).repeat(T, 1, 1)

        recon_next_obs = self.reconstructor(torch.cat([obs, action, p], -1))
        # recon_next_obs = self.reconstructor(torch.cat([obs, action], -1))

        recon_loss = 0.1 * (recon_next_obs - next_obs)**2

        self.optimizer.zero_grad()
        total_loss = recon_loss.mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.domain_learner.parameters(), max_norm = 0.5)
        torch.nn.utils.clip_grad_norm_(self.reconstructor.parameters(), max_norm = 0.5)
        self.optimizer.step()

        training_log_info = {}
        training_log_info['total_loss'] = total_loss.item()
        training_log_info['lr'] = self.optimizer.param_groups[0]['lr']

        return training_log_info


    def save(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        state_dict = {
            'domain_learner' : self.domain_learner.state_dict(),
            'reconstructor' : self.reconstructor.state_dict(),
            'optimizer' : self.optimizer.state_dict()
        }
        torch.save(state_dict, path)


    def load(self, pkl_path):
        state_dict = torch.load(pkl_path, map_location = self.device)
        self.domain_learner.load_state_dict(state_dict['domain_learner'])
        self.reconstructor.load_state_dict(state_dict['reconstructor'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.optimizer.param_groups[0]['lr'] = self.optim_params['lr']
        print('Load from %s successfully!' % pkl_path)



if __name__ == '__main__':
    env_params = {
        'env_id' : None,
        'env_class' : HalfCheetahEnv_RandomDynamics,
        'n_envs' : 4
    }
    model_params = {
        'z_dim' : 64,
        'hidden_dims' : [128]
    }
    optim_params = {
        'lr' : 0.001
    }
    training_params = {
        'batch_size' : 4,
        'is_episodic' : True
    }
    trainer = DeepsetsLatentTrainer(env_params, model_params, optim_params, training_params, 'auto')
    # trainer.load('./latent/model.pkl')
    # trainer.visualize_gradient_flow()

    # target_dir = './deepset_learner/'
    # trainer.learn(200, eval_frequency = None, log_path = target_dir + 'log/', best_model_path = None)
    # trainer.save(target_dir + 'model.pkl')


    trainer.load('deepset_learner_rd/model.pkl')
    device = trainer.device
    obs = trainer.env.reset()

    with torch.no_grad():
        for e in range(1000):
            action = trainer.env.random_action()

            masked_next_obs, masked_rewards, dones = trainer.env.step(action)
            trainer.buffer.add(
                obs = obs,
                next_obs = masked_next_obs,
                action = action,
                reward = masked_rewards,
                done = dones
            )

            obs = masked_next_obs
            if trainer.env.all_done():
                break

        obs, next_obs, action, _, _ = next(trainer.buffer.generate_episodic_data(4, device))

        ps = [trainer.domain_learner(torch.cat([obs[t], next_obs[t], action[t]], -1)) for t in range(1000)]
        p = torch.stack(ps).mean(0)
        p = p.detach().cpu().numpy()

        p01 = np.linalg.norm(p[0] - p[1])
        p02 = np.linalg.norm(p[0] - p[2])
        p03 = np.linalg.norm(p[0] - p[3])
        p12 = np.linalg.norm(p[1] - p[2])
        p13 = np.linalg.norm(p[1] - p[3])
        p23 = np.linalg.norm(p[2] - p[3])

        print(p01, p02, p03)
        print(p12, p13)
        print(p23)