import numpy as np
import torch
import random
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from common import np_to_tensor


def create_rnn_state(lstm_type, recurrent_num_layers, recurrent_hidden_dim, batch_size = 1, device = torch.device('cuda')):
    recurrent_hidden_shape = (recurrent_num_layers, batch_size, recurrent_hidden_dim)

    if lstm_type == 'lstm':
        return (torch.zeros(recurrent_hidden_shape).to(device), torch.zeros(recurrent_hidden_shape).to(device))
    elif lstm_type == 'gru':
        return torch.zeros(recurrent_hidden_shape).to(device)
    else:
        raise ValueError



class BaseEpisodeBuffer(object):
    def __init__(self, n_envs, buffer_size, obs_dim, act_dim) -> None:
        self.obs = np.zeros((buffer_size, n_envs, obs_dim), dtype = np.float32)
        self.next_obs = np.zeros((buffer_size, n_envs, obs_dim), dtype = np.float32)
        self.actions = np.zeros((buffer_size, n_envs, act_dim), dtype = np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype = np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype = np.float32)
        
        self.pos = 0
        self.capacity = buffer_size
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim


    def reset(self):
        self.obs.fill(0)
        self.next_obs.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.pos = 0


    def add(self, obs, next_obs, action, reward, done):
        if self.pos >= self.capacity:
            raise InterruptedError

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1


    def generate_episodic_data(self, batch_size, device):
        obs_ts = np_to_tensor(self.obs[:self.pos], device)
        next_obs_ts = np_to_tensor(self.next_obs[:self.pos], device)
        actions_ts = np_to_tensor(self.actions[:self.pos], device)
        rewards_ts = np_to_tensor(self.rewards[:self.pos], device)
        dones_ts = np_to_tensor(self.dones[:self.pos], device)

        sampler = BatchSampler(SubsetRandomSampler(range(self.n_envs)), batch_size, False)
        for env_indices in sampler:
            yield obs_ts[:, env_indices], \
                next_obs_ts[:, env_indices], \
                actions_ts[:, env_indices], \
                rewards_ts[:, env_indices], \
                dones_ts[:, env_indices]


    def generate_transition_data(self, batch_size, device):
        obs_ts = np_to_tensor(self.obs[:self.pos], device).view(-1, self.obs_dim)
        next_obs_ts = np_to_tensor(self.next_obs[:self.pos], device).view(-1, self.obs_dim)
        actions_ts = np_to_tensor(self.actions[:self.pos], device).view(-1, self.act_dim)
        rewards_ts = np_to_tensor(self.rewards[:self.pos], device).view(-1)
        dones_ts = np_to_tensor(self.dones[:self.pos], device).view(-1)

        sampler = BatchSampler(SubsetRandomSampler(range(obs_ts.size(0))), batch_size, False)
        for indices in sampler:
            yield obs_ts[indices], \
                next_obs_ts[indices], \
                actions_ts[indices], \
                rewards_ts[indices], \
                dones_ts[indices]





# class RolloutBuffer(object):
#     def __init__(self, n_envs, buffer_size, obs_dim, act_dim, gae_lambda = 0.95, gamma = 0.99) -> None:
#         self.gae_lambda = gae_lambda
#         self.gamma = gamma
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim

#         self.obs = np.zeros((buffer_size, n_envs, obs_dim), dtype = np.float32)
#         self.next_obs = np.zeros((buffer_size, n_envs, obs_dim), dtype = np.float32)
#         self.actions = np.zeros((buffer_size, n_envs, act_dim), dtype = np.float32)
#         self.rewards = np.zeros((buffer_size, n_envs), dtype = np.float32)
#         self.dones = np.zeros((buffer_size, n_envs), dtype = np.float32)
#         self.values = np.zeros((buffer_size, n_envs), dtype = np.float32)
#         self.action_logprobs = np.zeros((buffer_size, n_envs), dtype = np.float32)

#         self.pos = 0
#         self.buffer_size = buffer_size
#         self.n_envs = n_envs


#     def reset(self):
#         self.obs.fill(0)
#         self.next_obs.fill(0)
#         self.actions.fill(0)
#         self.rewards.fill(0)
#         self.dones.fill(0)
#         self.values.fill(0)
#         self.action_logprobs.fill(0)
#         self.pos = 0


#     def add(self, ob, next_ob, action, reward, done, value, action_logprob):
#         self.obs[self.pos] = ob
#         self.next_obs[self.pos] = next_ob
#         self.actions[self.pos] = action
#         self.rewards[self.pos] = reward
#         self.dones[self.pos] = done
#         self.values[self.pos] = value
#         self.action_logprobs[self.pos] = action_logprob
#         self.pos += 1


#     def finish_epoch(self, last_values, last_dones, env_total_steps = None):
#         # Compute returns, advantages and trajectory masks

#         self.advs = np.zeros(self.values.shape)
#         last_gae_lam = 0

#         for s in reversed(range(self.pos)):
#             if s == self.pos - 1:
#                 next_not_done = 1.0 - last_dones
#                 next_values = last_values
#             else:
#                 next_not_done = 1.0 - self.dones[s + 1]
#                 next_values = self.values[s + 1]

#             delta = self.rewards[s] + self.gamma * next_values * next_not_done - self.values[s]
#             last_gae_lam = delta + self.gamma * self.gae_lambda * next_not_done * last_gae_lam
#             self.advs[s] = last_gae_lam

#         self.returns = self.advs + self.values
#         self.traj_masks = 1 - self.dones
#         self.env_total_steps = env_total_steps


#     def generate_episodic_data(self, batch_size, device):
#         obs_ts = np_to_tensor(self.obs[:self.pos], device)
#         next_obs_ts = np_to_tensor(self.next_obs[:self.pos], device)
#         actions_ts = np_to_tensor(self.actions[:self.pos], device)
#         action_logprobs_ts = np_to_tensor(self.action_logprobs[:self.pos], device)
#         returns_ts = np_to_tensor(self.returns[:self.pos], device)
#         advs_ts = np_to_tensor(self.advs[:self.pos], device)
#         traj_masks_ts = np_to_tensor(self.traj_masks[:self.pos], device)

#         sampler = BatchSampler(SubsetRandomSampler(range(self.n_envs)), batch_size, False)
#         for env_indices in sampler:
#             yield obs_ts[:, env_indices], \
#                 next_obs_ts[:, env_indices], \
#                 actions_ts[:, env_indices, ...], \
#                 action_logprobs_ts[:, env_indices], \
#                 returns_ts[:, env_indices], \
#                 advs_ts[:, env_indices], \
#                 traj_masks_ts[:, env_indices]


#     def generate_sequential_data(self, seq_len, num_sequential_samples, device):
#         obs_ts = np_to_tensor(self.obs[:self.pos], device)
#         next_obs_ts = np_to_tensor(self.next_obs[:self.pos], device)
#         actions_ts = np_to_tensor(self.actions[:self.pos], device)
#         action_logprobs_ts = np_to_tensor(self.action_logprobs[:self.pos], device)
#         returns_ts = np_to_tensor(self.returns[:self.pos], device)
#         advs_ts = np_to_tensor(self.advs[:self.pos], device)
#         traj_masks_ts = np_to_tensor(self.traj_masks[:self.pos], device)

#         for _ in range(num_sequential_samples):
#             batch_obs_ts = []
#             batch_next_obs_ts = []
#             batch_actions_ts = []
#             batch_action_logprobs_ts = []
#             batch_returns_ts = []
#             batch_advs_ts = []
#             batch_traj_masks_ts = []
#             for e in range(self.n_envs):
#                 start_idx = np.random.randint(0, self.env_total_steps[e] - seq_len if self.env_total_steps[e] > seq_len else 1)
#                 indices = np.arange(start_idx, start_idx + seq_len)
#                 batch_obs_ts.append(obs_ts[indices, e])
#                 batch_next_obs_ts.append(next_obs_ts[indices, e])
#                 batch_actions_ts.append(actions_ts[indices, e])
#                 batch_action_logprobs_ts.append(action_logprobs_ts[indices, e])
#                 batch_returns_ts.append(returns_ts[indices, e])
#                 batch_advs_ts.append(advs_ts[indices, e])
#                 batch_traj_masks_ts.append(traj_masks_ts[indices, e])
#             yield torch.stack(batch_obs_ts, 1), \
#                 torch.stack(batch_next_obs_ts, 1), \
#                 torch.stack(batch_actions_ts, 1), \
#                 torch.stack(batch_action_logprobs_ts, 1), \
#                 torch.stack(batch_returns_ts, 1), \
#                 torch.stack(batch_advs_ts, 1), \
#                 torch.stack(batch_traj_masks_ts, 1)


#     def generate_transition_data(self, batch_size, device):
#         obs_ts = np_to_tensor(self.obs[:self.pos], device).view(-1, self.obs_dim)
#         next_obs_ts = np_to_tensor(self.next_obs[:self.pos], device).view(-1, self.obs_dim)
#         actions_ts = np_to_tensor(self.actions[:self.pos], device).view(-1, self.act_dim)
#         action_logprobs_ts = np_to_tensor(self.action_logprobs[:self.pos], device).view(-1)
#         returns_ts = np_to_tensor(self.returns[:self.pos], device).view(-1)
#         advs_ts = np_to_tensor(self.advs[:self.pos], device).view(-1)
#         traj_masks_ts = np_to_tensor(self.traj_masks[:self.pos], device).view(-1)

#         sampler = BatchSampler(SubsetRandomSampler(range(obs_ts.size(0))), batch_size, False)
#         for indices in sampler:
#             yield obs_ts[indices], next_obs_ts[indices], actions_ts[indices], action_logprobs_ts[indices], returns_ts[indices], advs_ts[indices], traj_masks_ts[indices]



if __name__ == '__main__':
    buf = RolloutBuffer(4, 100, 8, 3)
    for _ in range(100):
        buf.add(np.ones((4, 8)), np.ones((4, 8)), np.ones((4,3)), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
    steps = np.random.randint(0, 100, size = (4,))
    print(steps)
    buf.finish_epoch(np.zeros(4), np.zeros(4), steps)
    for data in buf.generate_sequential_data(10, 2, torch.device('cpu')):
        print(data[0].size())