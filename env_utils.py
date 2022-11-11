import gym
import numpy as np
from gym.spaces import Box


class DummyVecEnv(object):
    def __init__(self, env_id = '', env_class = None, n_envs = 4) -> None:
        # Parameters
        self.n_envs = n_envs

        # Environment and spaces
        self.vec_envs = [env_class() if env_class else gym.make(env_id) for _ in range(n_envs)]

        self.obs_space = self.vec_envs[0].observation_space
        self.obs_shape = self.obs_space.shape if isinstance(self.obs_space, Box) else (self.obs_space.n,)
        self.obs_dim = np.prod(self.obs_shape)

        self.act_space = self.vec_envs[0].action_space
        self.act_shape = self.act_space.shape if isinstance(self.act_space, Box) else (self.act_space.n,)
        self.act_dim = np.prod(self.act_shape)
        
        self.env_max_steps = self.vec_envs[0]._max_episode_steps if hasattr(self.vec_envs[0], '_max_episode_steps') else 1000

        # Others
        self.vec_total_rewards = np.zeros((self.n_envs), dtype = np.float32)
        self.vec_total_steps = np.zeros((self.n_envs), dtype = np.float32)
        self.vec_active = np.ones((self.n_envs), dtype = bool)


    def reset(self):
        self.vec_total_rewards.fill(0)
        self.vec_total_steps.fill(0)
        self.vec_active.fill(True)

        vec_obs = []
        for _, env in enumerate(self.vec_envs):
            vec_obs.append(env.reset())

        return np.stack(vec_obs)


    def all_done(self):
        return np.all(self.vec_active == False)


    def random_action(self):
        return np.array([self.act_space.sample() for _ in range(self.n_envs)])


    def step(self, action):
        '''
            Input:
                action: [n_envs, act_dim]
            Output:
                masked_next_obs: [n_envs, obs_shape]
                masked_rewards: [n_envs]
                dones: [n_envs]
        '''
        vec_obs = np.zeros((self.n_envs, *self.obs_shape))
        vec_rewards = np.zeros((self.n_envs))
        vec_dones = np.zeros((self.n_envs))

        for idx, a in enumerate(action):
            if self.vec_active[idx]:
                vec_obs[idx], vec_rewards[idx], vec_dones[idx], _ = self.vec_envs[idx].step(a)
                self.vec_total_rewards[idx] += vec_rewards[idx]
                self.vec_total_steps[idx] += 1
                self.vec_active[idx] = not vec_dones[idx]

        return vec_obs, vec_rewards, vec_dones



if __name__ == '__main__':
    n_envs = 5
    envs = DummyVecEnv('Ant-v4', None, n_envs)

    # for env in envs.vec_envs:
    #     print(env.model.body('bthigh').mass)
    
    obs = envs.reset()
    counter = 0
    while not envs.all_done():
        actions = np.array([envs.act_space.sample() for _ in range(n_envs)])
        next_obs, rewards, dones = envs.step(actions)
        # print(counter, dones)
        # print('----')
        # print(obs)
        # print(actions)
        # print(next_obs)
        # print(rewards)
        # print(dones)
        obs = next_obs
        counter += 1
    print(envs.vec_total_steps)

    # for env in envs.vec_envs:
    #     print(env.model.body('bthigh').mass)