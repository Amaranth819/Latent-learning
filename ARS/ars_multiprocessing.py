"""
Augmented Random Search:
    https://proceedings.neurips.cc/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf
Online mean/variance calculation:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""
import itertools as it
import pickle as pk
import time
import yaml
import os
import numpy as np
import gym
import multiprocessing
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tensorboardX import SummaryWriter


def augmented_random_search(tuple_args):
    env, M, delta, mean, var, gamma = tuple_args
    T = env._max_episode_steps if hasattr(env, '_max_episode_steps') else 1000
    delta_M = np.stack((M + gamma*delta, M - gamma*delta), axis=1)
    N = delta.shape[0]

    # Allocate arrays for reward and episode length
    r = np.zeros((N, 2, T))
    alive = np.zeros((N, 2))
    all_rollout_x = []


    for (k, s) in it.product(range(N), range(2)):
        # Reset environment for current rollout
        x = env.reset()
        
        for t in range(T):
            alive[k,s] = t # update episode length

            # Apply linear policy to standardized observation and perform action
            a = delta_M[k,s] @ ((x - mean) / np.sqrt(var + 1e-8))
            x, r[k,s,t], done, _ = env.step(a)
            all_rollout_x.append(x)

            if done:
                break

    return all_rollout_x, r, alive



def multiprocessing_augmented_random_search(
    env_name,
    N = 64,
    b = 20,
    alpha = 0.015,
    gamma = 0.025,
    num_updates = 500,
    resume_filename = None,
    save_root_path = './ars/',
    num_workers = 8,
):
    assert N % num_workers == 0

    envs = [gym.make(env_name) for _ in range(num_workers)]
    p = envs[0].action_space.shape[0]
    n = envs[0].observation_space.shape[0]

    if resume_filename:
        # Load progress
        with open(resume_filename, "rb") as f: 
            (metrics, M, mean, var, nx) = pk.load(f)
        print('Load from %s!' % resume_filename)
    else:
        # Initialize linear policy matrix and observation statistics
        M = np.zeros((p, n))
        mean = np.zeros(n)
        var = np.ones(n)
        nx = 0 # number of samples for online mean/variance calculation        
        metrics = {key: [] for key in ('runtime','lifetime','reward')} # Initialize learning metrics
        print('Train a new model!')

    # Updates so far
    E = len(metrics['runtime'])

    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)

    log = SummaryWriter(os.path.join(save_root_path + 'log/'))

    with multiprocessing.Pool(num_workers) as pool:
        for e in np.arange(num_updates) + 1:
            update_start = time.perf_counter() # time the update
            last_nx = nx

            # Sample random perturbations to linear policy
            delta = np.random.randn(N, p, n).reshape(num_workers, -1, p, n)

            worker_args = [(envs[i], M, delta[i], mean, var, gamma) for i in range(num_workers)]
            results = pool.map(augmented_random_search, worker_args)
            all_rollout_xs, rs, alives = zip(*results)

            all_rollout_xs = np.concatenate(all_rollout_xs, 0) # [N*2*T, n]
            rs = np.concatenate(rs, 0) # [N, 2, T]
            alives = np.concatenate(alives, 0) # [N, 2]
            delta = delta.reshape(N, p, n)

            # Policy update rule
            rs = rs.sum(axis=2)
            kappa = np.argsort(rs.max(axis=1))[-b:]
            drs = (rs[kappa,0]-rs[kappa,1]).reshape(-1, 1, 1)
            sigma_R = rs[kappa].std()
            M = M + alpha / sigma_R * np.mean(drs * delta[kappa], axis=0)
            curr_nx = all_rollout_xs.shape[0]
            nx += curr_nx

            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance: Parallel variance
            avg_curr_rollout_x = np.mean(all_rollout_xs, 0)
            M_prev = var * last_nx
            M_curr = np.std(all_rollout_xs, 0)**2 * curr_nx
            M2 = M_prev + M_curr + (avg_curr_rollout_x - mean) ** 2 * last_nx * curr_nx / nx
            var = M2 / nx
            mean = (mean * last_nx + avg_curr_rollout_x * curr_nx) / nx

            # Update learning metrics
            metrics['runtime'].append(time.perf_counter() - update_start)
            metrics['lifetime'].append(alives.mean())
            metrics['reward'].append(rs.mean())

            log.add_scalar('Lifetime', alives.mean(), e)
            log.add_scalar('Reward', rs.mean(), e)

            # Print progress update
            print(f"update {E+e}/{E+num_updates}: reward ~ {metrics['reward'][-1]:.2f}, " + \
                f"|μ| ~ {np.fabs(mean).mean():.2f} (nx={nx}), " + \
                f"|Σ < ∞|={(var < np.inf).sum()}, |Σ| ~ {np.fabs(var[var < np.inf]).mean():.2f}, " + \
                f"T ~ {metrics['lifetime'][-1]:.2f}, " + \
                f"Running time ~ {metrics['runtime'][-1]:.2f}s")

            # Save progress
            with open(os.path.join(save_root_path, 'ars.pkl'), "wb") as f: 
                pk.dump((metrics, M, mean, var, nx), f)

    log.close()

    # Return final metrics and policy
    return (metrics, M, mean, var, nx)


def viz(env_name, fname = 'ars.pkl'):
    with open(fname, 'rb') as f:
        (metrics, M, mean, var, nx) = pk.load(f)

    env = gym.make(env_name)
    x = env.reset()
    tr = 0
    steps = 0
    done = False
    recoder = VideoRecorder(env, 'ars.mp4', enabled = True)

    while not done:
        a = M @ ((x - mean) / np.sqrt(var + 1e-8))
        x, r, done, _ = env.step(a)
        recoder.capture_frame()

        steps += 1
        tr += r

    recoder.close()
    recoder.enabled = False
    env.close()
    print('Steps = %.2f | Reward = %.2f' % (steps, tr))


if __name__ == '__main__':
    params = dict(
        env_name = 'Humanoid-v4',
        N = 480,
        b = 240,
        alpha = 0.02,
        gamma = 0.0075,
        num_updates = 500,
        resume_filename = None,
        save_root_path = './ars/',
        num_workers = 16
    )

    print(params)

    with open('config.yaml', 'w') as f:
        yaml.dump(params, f)

    multiprocessing_augmented_random_search(**params)

    viz(env_name = params['env_name'], fname = os.path.join(params['save_root_path'], 'ars.pkl'))