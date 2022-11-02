import gym
import pickle
import itertools
import numpy as np
import time
from tensorboardX import SummaryWriter

'''
    In rollout phase, each worker does rollouts for N/num_workers signed perturbations.
'''
class SingleARSWorker(object):
    def __init__(
        self, 
        env_name, 
        # alpha = 0.015, 
        # gamma = 0.025, 
        # N = 4, 
        # b = 20, 
        # num_workers = 16,
    ):
        '''
            epochs: number of updating the model
            alpha: step-size
            gamma: standard deviation of the exploration noise
            N: number of directions sampled per iteration
            b: number of top-performing directions
            # num_workers: number of CPUs
        '''
        self.env = gym.make(env_name)
        self.p = self.env.action_space.shape[0]
        self.n = self.env.observation_space.shape[0]
        self.T = self.env._max_episode_steps if hasattr(self.env, '_max_episode_steps') else 1000


    def one_epoch_rollout(self, M, mu, sigma, nx, gamma, N):
        # Sample random perturbations to linear policy
        delta = np.random.randn(N, self.p, self.n)
        delta_M = np.stack((M + gamma * delta, M - gamma * delta), axis = 1)

        # Allocate arrays for reward and episode length
        r = np.zeros((N, 2, self.T))
        alive = np.zeros((N, 2))

        # Rollouts for each signed perturbation
        for (k, s) in itertools.product(range(N), range(2)):

            # Reset environment for current rollout
            x = self.env.reset()
            for t in range(self.T):
                alive[k,s] = t # update episode length

                # update observation statistics
                dx = x - mu
                mu += dx / (nx + 1)
                sigma = (sigma * nx + dx * (x - mu)) / (nx + 1)
                nx += 1

                # Apply linear policy to standardized observation and perform action
                a = delta_M[k,s] @ ((x - mu) / np.sqrt(np.where(sigma < 1e-8, np.inf, sigma)))
                x, r[k,s,t], done, _ = self.env.step(a)
                if done: break # stop episode if done early

        return r, alive, delta



class MultiprocessARS(object):
    def __init__(
        self, 
        env_name, 
        alpha = 0.015, 
        gamma = 0.025, 
        N = 64, 
        b = 20, 
        num_workers = 8
    ) -> None:
        self.workers = [SingleARSWorker(env_name) for _ in range(num_workers)]

        # Initialize linear policy matrix and observation statistics
        env = gym.make(env_name)
        p = env.action_space.shape[0]
        n = env.observation_space.shape[0]
        self.M = np.zeros((p, n))
        self.mu = np.zeros(n)
        self.sigma = np.ones(n)
        self.nx = 0 # number of samples for online mean/variance calculation
        self.metrics = {key: [] for key in ('runtime','lifetime','reward')} # Initialize learning metrics

        self.alpha = alpha
        self.gamma = gamma
        self.N = N
        self.b = b


    def load(self, path):
        with open(path, 'rb') as f:
            (self.metrics, self.M, self.mu, self.sigma, self.nx) = pickle.load(f)


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.metrics, self.M, self.mu, self.sigma, self.nx), f)


    def train(self, epochs = 500, fname = 'ars.pkl', log_path = './log/'):
        # Updates so far
        J = len(self.metrics['runtime'])

        log = SummaryWriter(log_path)

        for j in np.arange(epochs) + 1:
            update_start = time.perf_counter() # time the update

            # Rollouts for each worker
            rs, alives, deltas = zip(*[worker.one_epoch_rollout() for worker in self.workers])
            rs = np.concatenate(rs, axis = 0)
            alives = np.concatenate(alives, axis = 0)
            deltas = np.concatenate(deltas, axis = 0)

            # Policy update rule
            rs = np.sum(rs, axis = 2)
            kappa = np.argsort(np.max(rs, axis = 1))[-self.b:]
            drs = (rs[kappa, 0] - rs[kappa, 1]).reshape(-1, 1, 1)
            alpha_R = np.std(rs[kappa])
            self.M = self.M + self.alpha / alpha_R * np.mean(drs * deltas[kappa], axis = 0)

            # Update learning metrics
            self.metrics['runtime'].append(time.perf_counter() - update_start)
            self.metrics['lifetime'].append(alives.mean())
            self.metrics['reward'].append(rs.mean())

            # Logging
            log.add_scalar('Lifetime', alives.mean(), j)
            log.add_scalar('Reward', rs.mean(), j)

            # Print progress update
            print(f"update {J+j}/{J+epochs}: reward ~ {self.metrics['reward'][-1]:.2f}, " + \
                f"|μ| ~ {np.fabs(self.mu).mean():.2f} (nx={self.nx}), " + \
                f"|Σ < ∞|={(self.sigma < np.inf).sum()}, |Σ| ~ {np.fabs(self.sigma[self.sigma < np.inf]).mean():.2f}, " + \
                f"T ~ {self.metrics['lifetime'][-1]:.2f} " + \
                f"[{self.metrics['reward'][-1]:.2f}s]")

            # Save progress
            self.save(fname)

        # Shut down
        log.close()