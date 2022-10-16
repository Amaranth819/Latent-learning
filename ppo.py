import os
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from logger import Logger
from data_buffer import RolloutBuffer
from env_utils import DummyVecEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from common import np_to_tensor, tensor_to_np, get_device
from collections import defaultdict
from worldmodel_a2c import WorldModelA2C



class PPO(object):
    def __init__(
        self, 
        env_id = '', 
        env_class = None,
        n_envs = 8,
        actor_lr = 0.0003,
        critic_lr = 0.001,
        worldmodel_lr = 0.0001,
        num_gru_layers = 2,
        state_dim = 128,
        hidden_dim = 256,
        min_state_std = 0.1,
        max_state_std = 1.0,
        fixed_action_std = 0.5,
        batch_size = 16,
        seq_len = 25,
        collect_intervals = 20,
        repeat_batch = 2,
        epsilon = 0.2,
        gamma = 0.99,
        gae_lambda = 0.95,
        vloss_coef = 0.1,
        recon_obs_loss_coef = 0.1,
        clip_loss_coef = 100.0,
        actor_entropy_coef = 10.0,
        device = 'auto'
    ) -> None:
        self.env_id = env_id
        self.env_class = env_class
        self.epsilon = epsilon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.worldmodel_lr = worldmodel_lr
        self.num_gru_layers = num_gru_layers
        self.min_state_std = min_state_std
        self.max_state_std = max_state_std
        self.state_dim = state_dim
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.collect_intervals = collect_intervals
        self.repeat_batch = repeat_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.hidden_dim = hidden_dim
        self.fixed_action_std = fixed_action_std
        self.vloss_coef = vloss_coef
        self.recon_obs_loss_coef = recon_obs_loss_coef
        self.clip_loss_coef = clip_loss_coef
        self.actor_entropy_coef = actor_entropy_coef

        # Environment
        self.training_env = DummyVecEnv(env_id, env_class, n_envs)
        
        # NN
        self.device = get_device(device)

        self.actor_critic = WorldModelA2C(
            obs_dim = self.training_env.obs_dim,
            act_dim = self.training_env.act_dim,
            state_dim = self.state_dim,
            hidden_dim = self.hidden_dim,
            num_gru_layers = self.num_gru_layers,
            min_state_std = self.min_state_std,
            max_state_std = self.max_state_std,
            fixed_action_std = self.fixed_action_std
        ).to(self.device)
        actor_critic_params = self.actor_critic.get_parameter_dict()
        self.optimizer = torch.optim.Adam([
            {'params' : actor_critic_params['actor'], 'lr' : actor_lr},
            {'params' : actor_critic_params['critic'], 'lr' : critic_lr},
            {'params' : actor_critic_params['worldmodel'], 'lr' : worldmodel_lr},
        ])

        # Buffer
        self.buffer = RolloutBuffer(
            n_envs,
            self.training_env.env_max_steps,
            self.training_env.obs_dim,
            self.training_env.act_dim,
            gae_lambda,
            gamma
        )

    
    def collect(self, deterministic = False):
        (h_t, s_t) = self.actor_critic.rssm.init_states(self.training_env.n_envs)
        obs = self.training_env.reset()
        self.buffer.reset()
        a_t = torch.zeros((self.training_env.n_envs, self.training_env.act_dim)).to(h_t.device)

        with torch.no_grad():
            counter = 0

            while not self.training_env.all_done() and counter < self.training_env.env_max_steps:
                obs_ts = np_to_tensor(obs, self.device)
                _, a_t, action_logprobs_ts, value_ts, (h_t, s_t) = self.actor_critic.step(obs_ts, a_t, h_t, s_t, deterministic)
                
                actions = tensor_to_np(a_t)
                next_obs, rewards, dones = self.training_env.step(actions)

                env_mask = self.training_env.vec_active
                masked_actions = actions * env_mask[:, None]
                masked_values = tensor_to_np(value_ts) * env_mask
                masked_action_logprobs = tensor_to_np(action_logprobs_ts) * env_mask

                self.buffer.add(
                    obs, next_obs, masked_actions, rewards, dones, masked_values, masked_action_logprobs
                )

                obs = next_obs
                counter += 1

            # Compute returns and advantages
            obs_ts = np_to_tensor(obs, self.device)
            _, _, _, value_ts, (_, _) = self.actor_critic.step(obs_ts, a_t, h_t, s_t)
            self.buffer.finish_epoch(tensor_to_np(value_ts), 1 - self.training_env.vec_active, self.training_env.vec_total_steps)

        return np.mean(self.training_env.vec_total_rewards), \
            np.std(self.training_env.vec_total_rewards), \
            np.mean(self.training_env.vec_total_steps), \
            np.std(self.training_env.vec_total_steps)


    def update(self, batch_obs, batch_next_obs, batch_actions, batch_old_action_logprobs, batch_returns, batch_advs, batch_traj_masks):
        features, priors, posteriors, batch_recon_obs_logprob, batch_new_action_logprob, batch_action_dist_entropy, batch_v_logprob = self.actor_critic(batch_next_obs, batch_actions, batch_returns)

        # Clip loss
        norm_advs = (batch_advs - batch_advs.mean()) / (batch_advs.std() + 1e-8)
        ratio = (batch_new_action_logprob - batch_old_action_logprobs).exp()
        surr1 = ratio * norm_advs
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * norm_advs
        clip_loss = self.clip_loss_coef * -(torch.min(surr1, surr2) * batch_traj_masks).mean()

        # Entropy loss
        entropy_loss = self.actor_entropy_coef * -batch_action_dist_entropy.mean()

        # KL divergence loss
        priors_dist = self.actor_critic.rssm.create_diag_normal(priors)
        posteriors_dist = self.actor_critic.rssm.create_diag_normal(posteriors)
        kl_div_loss = torch.distributions.kl.kl_divergence(priors_dist, posteriors_dist).sum(0).mean()

        # Reconstruction observation loss
        recon_loss = self.recon_obs_loss_coef * -batch_recon_obs_logprob.sum(0).mean()

        # Critic loss
        critic_loss = self.vloss_coef * -batch_v_logprob.sum(0).mean()

        # Optimize
        self.optimizer.zero_grad()
        total_loss = clip_loss + entropy_loss + kl_div_loss + recon_loss + critic_loss
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm = 0.05)
        self.optimizer.step()

        return {
            'Total loss' : total_loss.item(),
            'Clip loss' : clip_loss.item(),
            'Entropy loss' : entropy_loss.item(),
            'KL div loss' : kl_div_loss.item(),
            'Recon next obs loss' : recon_loss.item(),
            # 'Actor loss' : clip_loss.item() + entropy_loss.item(),
            'Critic loss' : critic_loss.item(),
            'Actor LR' : self.optimizer.param_groups[0]['lr'],
            'Critic LR' : self.optimizer.param_groups[1]['lr'],
            'WorldModel LR' : self.optimizer.param_groups[2]['lr']
        }


    def one_epoch(self):
        training_info = defaultdict(lambda: [])

        # Sampling trajectory
        sample_rewards_mean, _, sample_steps_mean, _ = self.collect(False)
        training_info['Sample Rewards Mean'] = sample_rewards_mean
        training_info['Sample Steps Mean'] = sample_steps_mean

        # Training
        for _ in range(self.repeat_batch):
            for batch_obs_ts, batch_next_obs_ts, batch_actions_ts, batch_action_logprobs_ts, batch_returns_ts, batch_advs_ts, batch_traj_masks_ts in self.buffer.generate_sequential_data(self.seq_len, self.collect_intervals, self.device):
                batch_training_info = self.update(batch_obs_ts, batch_next_obs_ts, batch_actions_ts, batch_action_logprobs_ts, batch_returns_ts, batch_advs_ts, batch_traj_masks_ts)

                for tag, val in batch_training_info.items():
                    training_info[tag].append(val)

        # Summary
        for tag, val_list in training_info.items():
            training_info[tag] = np.mean(val_list)

        return training_info


    def learn(self, epochs, eval_frequency = 10, log_path = None, best_model_path = None):
        if epochs <= 0:
            print('No training task!')
            return

        log = None if log_path is None else Logger(log_path)

        # Eval before training
        if eval_frequency:
            eval_rewards_mean, eval_rewards_std, eval_steps_mean, eval_steps_std = self.collect(True)
            if log is not None:
                log.add(0, {'Sample Rewards Mean' : eval_rewards_mean, 'Sample Steps Mean' : eval_steps_mean}, 'Eval/')
            print('Epoch 0 Eval: Rewards = %.2f +- %.2f | Steps = %.2f +- %.2f' % (eval_rewards_mean, eval_rewards_std, eval_steps_mean, eval_steps_std))


        for e in np.arange(epochs) + 1:
            training_info = self.one_epoch()

            if log is not None:
                log.add(e, training_info, 'Training/')

            print('####################')
            print('# Epoch: %d' % e)
            print('# Sampled episodes: %d' % (e * self.n_envs))
            for tag, scalar_val in training_info.items():
                print('# %s: %.5f' % (tag, scalar_val))
            print('####################\n')

            if eval_frequency is not None and e % eval_frequency == 0:
                curr_eval_rewards_mean, curr_eval_rewards_std, curr_eval_steps_mean, curr_eval_steps_std = self.collect(True)
                print('Epoch %d Eval: Rewards = %.2f +- %.2f | Steps = %.2f +- %.2f' % (e, curr_eval_rewards_mean, curr_eval_rewards_std, curr_eval_steps_mean, curr_eval_steps_std))
                if log is not None:
                    log.add(e, {'Sample Rewards Mean' : curr_eval_rewards_mean, 'Sample Steps Mean' : curr_eval_steps_mean}, 'Eval/')

                if curr_eval_rewards_mean > eval_rewards_mean:
                    print('Get a better model!\n')
                    eval_rewards_mean = curr_eval_rewards_mean
                    if best_model_path is not None:
                        self.save(best_model_path)
                else:
                    print('Don\'t get a better model!\n')



    def record_video(self, video_path):
        eval_env = self.env_class() if self.env_class else gym.make(self.env_id)
        obs = eval_env.reset()
        (h_t, s_t) = self.actor_critic.rssm.init_states(1)
        a_t = torch.zeros((1, self.training_env.act_dim)).to(h_t.device)
        done = False
        episode_rewards, episode_steps = 0, 0
        video_recorder = VideoRecorder(eval_env, video_path, enabled = True)
        max_episode_steps = eval_env._max_episode_steps if hasattr(eval_env, '_max_episode_steps') else 1000

        with torch.no_grad():
            while not done and episode_steps < max_episode_steps:
                obs_ts = np_to_tensor(obs, self.device).unsqueeze(0)
                # _, action_ts, _, rnn_state, _ = self.actor_critic(obs_ts, rnn_state, True)
                _, a_t, _, _, (h_t, s_t) = self.actor_critic.step(obs_ts, a_t, h_t, s_t, True)
                action = tensor_to_np(a_t.squeeze())
                next_obs, r, done, _ = eval_env.step(action)

                video_recorder.capture_frame()
                obs = next_obs
                episode_rewards += r
                episode_steps += 1

        print('Reward = %.3f | Step = %d' % (episode_rewards, episode_steps))

        video_recorder.close()
        video_recorder.enabled = False
        eval_env.close()


    def save(self, path = './PPO/PPO.pkl'):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        state_dict = {
            'worldmodela2c' : self.actor_critic.state_dict(),
            'optimizer' : self.optimizer.state_dict()
        }
        torch.save(state_dict, path)


    def load(self, PPO_pkl_path):
        state_dict = torch.load(PPO_pkl_path, map_location = self.device)
        self.actor_critic.load_state_dict(state_dict['worldmodela2c'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.optimizer.param_groups[0]['lr'] = self.actor_lr
        self.optimizer.param_groups[1]['lr'] = self.critic_lr
        self.optimizer.param_groups[2]['lr'] = self.worldmodel_lr
        print('Load from %s successfully!' % PPO_pkl_path)



'''
    Visualize gradient flow
'''
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
    plt.ylim(bottom = -0.001) # zoom in on the lower gradient regions
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
    algo = PPO('HalfCheetah-v4', n_envs = 4)
    algo.one_epoch()
    plot_grad_flow(algo.actor_critic.cpu().named_parameters())