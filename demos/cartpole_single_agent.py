from spurl.algorithms.reinforce.discrete import REINFORCE
from spurl.core import train, test
from spurl.utils import save_model, load_model, save_environment_render, build_policy_network

import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

tf.random.set_seed(42)
np.random.seed(42)

env = gym.make('CartPole-v1')

state_shape = env.observation_space.shape
action_space = env.action_space
num_actions = action_space.n

# Build policy network 
policy_network = build_policy_network(state_shape, 
                                      action_size = num_actions, 
                                      action_space = action_space, 
                                      policy_type = 'fcn',
                                      layers = [[], [64, 32, 14]])

reinforce = REINFORCE(env, policy_network, artificial_truncation=512)

# change some parameters of the algorithm
reinforce.gamma = 0.99
reinforce.learning_rate = 0.0001
reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, epsilon=1e-6, clipnorm=1e1)

reinforce = train(reinforce, trials=64, episodes_per_trial=8, epochs_per_trial=4, batch_size=32, verbose=True)

rewards, lengths = test(reinforce, trials=2, episodes_per_trial=8, deterministic=True)

# saving loading functionality
temp_path = r'./temp'
if not os.path.exists(temp_path):
    os.mkdir(temp_path)

save_model(reinforce, os.path.join(temp_path, 'model'))

del reinforce, policy_network, rewards, lengths

policy_network = None

reinforce = REINFORCE(env, policy_network, artificial_truncation=512)

reinforce = load_model(reinforce, os.path.join(temp_path, 'model'))

rewards, lengths = test(reinforce, trials=2, episodes_per_trial=8, deterministic=True)

del env

#render the environment
rendering_env = gym.make('CartPole-v1', render_mode='rgb_array')

save_environment_render(rendering_env, algorithm=reinforce, save_path=os.path.join(temp_path, 'cartpole_trajectory'), deterministic=True, artificial_truncation=512)



