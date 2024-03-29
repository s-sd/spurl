from spurl.algorithms.reinforce.continuous import REINFORCE
from spurl.utils import save_model, load_model, save_environment_render, build_policy_network
from spurl.core import train, test

import tensorflow as tf
import gymnasium as gym
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

# learn to always pick 0.5s
class DummyEnv(gym.Env):
    def __init__(self, action_size):
        self.episode_terminate = 256
        self.action_space = gym.spaces.Box(low=0, high=1, shape=action_size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(32, 32))
        self.step_num = 0
    def step(self, action):
        if (action - np.array([[0.5, 0.5],[0.5, 0.5]]) < 0.2).all():
            reward = 1
        elif (action - np.array([[0.5, 0.5],[0.5, 0.5]]) < 0.2).any():
            reward = 0.5
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

## An example of how to build policy networks from scratch 
# def build_policy_network(state_shape, action_size):
#     inputs = tf.keras.layers.Input(shape=state_shape)
#     flat = tf.keras.layers.Flatten()(inputs)
#     dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
#     dense2 = tf.keras.layers.Dense(np.prod(action_size), activation='linear')(dense1)
#     reshape = tf.keras.layers.Reshape(action_size)(dense2)
#     policy_network = tf.keras.Model(inputs=inputs, outputs=reshape)
#     return policy_network

action_size = (2, 2)

env = DummyEnv(action_size)

action_space = env.action_space
state_space = env.observation_space

policy_network = build_policy_network(state_space, 
                                      action_space,
                                      policy_type = 'fcn',
                                      layers = [128],
                                      activation_fn = 'linear')

reinforce = REINFORCE(env, policy_network, scale=0.2)

reinforce = train(reinforce, trials=12, episodes_per_trial=16, epochs_per_trial=1, batch_size=16, verbose=True)
    
history = test(reinforce, trials=2, episodes_per_trial=4, deterministic=True)
