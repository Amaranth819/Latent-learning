import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from logger import Logger
from env_utils import DummyVecEnv
from data_buffer import BaseEpisodeBuffer
from common import get_device, np_to_tensor, tensor_to_np
from collections import defaultdict


class BaseTrainer(object):
    def __init__(
        self,
        env_params,
        model_params,
        optim_params,
        training_params,
        device = 'auto'
    ) -> None:
        self.env_params = env_params
        self.model_params = model_params
        self.optim_params = optim_params
        self.training_params = training_params

        self.env = DummyVecEnv(
            env_params['env_id'],
            env_params['env_class'],
            env_params['n_envs']
        )
        self.device = get_device(device)
        self.buffer = BaseEpisodeBuffer(
            env_params['n_envs'],
            self.env.env_max_steps,
            self.env.obs_dim,
            self.env.act_dim
        )

        self.model = None
        self.optimizer = None
        self.lr_decay = None
        self.create_model(model_params, optim_params)
        if self.model is not None:
            self.init_model_weight(self.model)


    def create_model(self, model_params, optim_params):
        pass


    def init_model_weight(self, net):
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.uniform_(-0.1, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()


    def predict(self, obs_ts, deterministic = False):
        raise NotImplementedError


    def collect(self, model_prediction = True, deterministic = False):
        obs = self.env.reset()
        self.buffer.reset()
        
        with torch.no_grad():
            curr_steps = 0

            while not self.env.all_done() and curr_steps < self.env.env_max_steps:
                if model_prediction:
                    obs_ts = np_to_tensor(obs, self.device)
                    action_ts = self.predict(obs_ts, deterministic = deterministic)
                    action = tensor_to_np(action_ts)
                else:
                    action = self.env.sample_actions()

                masked_next_obs, masked_rewards, dones, masked_actions = self.env.step(action)
                self.buffer.add(
                    obs = obs,
                    next_obs = masked_next_obs,
                    action = masked_actions,
                    reward = masked_rewards,
                    done = dones
                )

                obs = masked_next_obs
                curr_steps += 1

        sample_episode_rewards_mean = np.mean(self.env.vec_total_rewards)
        sample_episode_rewards_std = np.std(self.env.vec_total_rewards)
        sample_episode_steps_mean = np.mean(self.env.vec_total_steps)
        sample_episode_steps_std = np.std(self.env.vec_total_steps)
        return (sample_episode_rewards_mean, sample_episode_rewards_std), (sample_episode_steps_mean, sample_episode_steps_std)


    def train(self, batch):
        raise NotImplementedError


    def one_epoch(self):
        log_info = defaultdict(lambda: [])

        # Sampling trajectory
        (sample_episode_rewards_mean, _), (sample_episode_steps_mean, _) = self.collect(False, False)
        log_info['Sample Rewards Mean'] = sample_episode_rewards_mean
        log_info['Sample Steps Mean'] = sample_episode_steps_mean

        # Training
        for batch in self.buffer.generate_transition_data(self.training_params['batch_size'], self.device):
            batch_training_log_info = self.train(batch)

            for tag, val in batch_training_log_info.items():
                log_info[tag].append(val)

        # Summary
        for tag, val_list in log_info.items():
            log_info[tag] = np.mean(val_list)

        return log_info


    def learn(self, epochs, eval_frequency = 10, log_path = None, best_model_path = None):
        if epochs <= 0:
            print('No training task!')
            return

        log = None if log_path is None else Logger(log_path)

        # Eval before training
        if eval_frequency:
            (eval_rewards_mean, eval_rewards_std), (eval_steps_mean, eval_steps_std) = self.collect(False, True)
            if log is not None:
                log.add(0, {'Sample Rewards Mean' : eval_rewards_mean, 'Sample Steps Mean' : eval_steps_mean}, 'Eval/')
            print('Epoch 0 Eval: Rewards = %.2f +- %.2f | Steps = %.2f +- %.2f' % (eval_rewards_mean, eval_rewards_std, eval_steps_mean, eval_steps_std))

        # Training
        for e in np.arange(epochs) + 1:
            training_info = self.one_epoch()

            if log is not None:
                log.add(e, training_info, 'Train/')

            print('####################')
            print('# Epoch: %d' % e)
            print('# Sampled episodes: %d' % (e * self.env_params['n_envs']))
            for tag, scalar_val in training_info.items():
                print('# %s: %.5f' % (tag, scalar_val))
            print('####################\n')

            if self.lr_decay is not None:
                self.lr_decay.step()

            # Evaluate every certain epochs
            if eval_frequency is not None and e % eval_frequency == 0:
                curr_eval_rewards_mean, curr_eval_rewards_std, curr_eval_steps_mean, curr_eval_steps_std = self.collect(False, True)
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


    def save(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        state_dict = {
            'model' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict()
        }
        torch.save(state_dict, path)


    def load(self, pkl_path):
        state_dict = torch.load(pkl_path, map_location = self.device)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.optimizer.param_groups[0]['lr'] = self.optim_params['lr']
        print('Load from %s successfully!' % pkl_path)


    '''
        Visualize gradient flow
    '''
    def visualize_gradient_flow(self):
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



        self.collect(False, False)
        batch = next(self.buffer.generate_transition_data(self.training_params['batch_size'], self.device))
        self.train(batch)
    
        plot_grad_flow(self.model.cpu().named_parameters())



if __name__ == '__main__':
    env_params = {
        'env_id' : 'CartPole-v1',
        'env_class' : None,
        'n_envs' : 2   
    }
    model_params = {}
    optim_params = {}
    training_params = {}
    trainer = BaseTrainer(env_params, model_params, optim_params, training_params, 'auto')
    trainer.collect(False, False)
    print(trainer.buffer.pos)
    for i in range(trainer.buffer.pos):
        print(i, trainer.buffer.obs[i], trainer.buffer.actions[i], trainer.buffer.next_obs[i], trainer.buffer.dones[i])