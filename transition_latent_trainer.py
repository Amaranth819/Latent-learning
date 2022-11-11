import torch
from network import TransitionLatentModel, BaseLatentModel
from base_trainer import BaseTrainer


class TransitionLatentTrainer(BaseTrainer):
    def __init__(self, env_params, model_params, optim_params, training_params, device='auto') -> None:
        super().__init__(env_params, model_params, optim_params, training_params, device)


    def create_model(self, model_params, optim_params):
        self.model = BaseLatentModel(
            obs_dim = self.env.obs_dim,
            z_dim = model_params['z_dim'],
            hidden_dims = model_params['hidden_dims']
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            params = self.model.parameters(),
            lr = optim_params['lr'],
            betas = [0.5, 0.999],
            eps = 1e-3
        )

        self.lr_decay = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = [40], gamma = 0.1)


    def predict(self, obs_ts, deterministic = False):
        pass


    def train(self, batch):
        obs, _, _, _, _ = batch

        (_, z_mu, z_std), obs_recon = self.model.forward(obs)
        kl = 0.5 * torch.sum(-2 * torch.log(z_std) + z_std.pow(2) + z_mu.pow(2) - 1., dim = -1)

        # obs_recon_logprob = -obs_recon_dist.log_prob(obs)
        obs_recon_loss = 0.1 * (obs_recon - obs).pow(2).sum(-1)

        self.optimizer.zero_grad()
        total_loss = (kl + obs_recon_loss).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 0.5)
        self.optimizer.step()

        training_log_info = {}
        training_log_info['kl'] = kl.mean().item()
        training_log_info['obs_recon_loss'] = obs_recon_loss.mean().item()
        training_log_info['total_loss'] = total_loss.item()
        training_log_info['lr'] = self.optimizer.param_groups[0]['lr']

        return training_log_info


if __name__ == '__main__':
    env_params = {
        'env_id' : 'HalfCheetah-v4',
        'env_class' : None,
        'n_envs' : 8
    }
    model_params = {
        'z_dim' : 64,
        'hidden_dims' : [128]
    }
    optim_params = {
        'lr' : 0.001
    }
    training_params = {
        'batch_size' : 128
    }
    trainer = TransitionLatentTrainer(env_params, model_params, optim_params, training_params, 'auto')
    # trainer.load('./latent/model.pkl')
    trainer.visualize_gradient_flow()

    # target_dir = './latent/'
    # trainer.learn(200, eval_frequency = None, log_path = target_dir + 'log/', best_model_path = None)
    # trainer.save(target_dir + 'model.pkl')

    # trainer.load('./latent/model.pkl')
    # trainer.learn(500, eval_frequency = None, log_path = './latent1/log/', best_model_path = None)
    # trainer.save('./latent1/model.pkl')