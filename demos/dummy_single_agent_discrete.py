from spurl.algorithms.reinforce.discrete import REINFORCE

from spurl.core import train, test
from spurl.utils import * 

import tensorflow as tf
import gymnasium as gym
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

# learn to always pick 0
class DummyEnv(gym.Env):
    def __init__(self, num_actions):
        self.episode_terminate = 256
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(32, 32))
        self.step_num = 0
    def step(self, action):
        if action == 0:
            reward = 1
        else:
            reward = 0
        
        if self.step_num >= self.episode_terminate-1:
            done = True
        else:
            done = False
        self.step_num += 1
        return np.random.rand(32, 32), reward, done, {}
    def reset(self):
        self.step_num = 0
        return np.random.rand(32, 32)

num_actions = 4

env = DummyEnv(num_actions)

# Building policy network
state_shape = env.observation_space.shape
action_space = env.action_space
output_shape = (action_space.n,)
policy_network = build_policy_network(state_shape,
                                      output_shape,
                                      action_space,
                                      policy_type = 'fcn',
                                      layers = [[],[128]])

reinforce = REINFORCE(env, policy_network)

reinforce = train(reinforce, trials=8, episodes_per_trial=8, epochs_per_trial=2, batch_size=16, verbose=True)

history = test(reinforce, trials=2, episodes_per_trial=4, deterministic=True)
