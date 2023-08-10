from spurl.algorithms.reinforce.discrete import REINFORCE
from spurl.core import train, test
from spurl.utils import save_model, load_model, save_environment_render

import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

tf.random.set_seed(42)
np.random.seed(42)

def build_policy_network(state_shape, num_actions):
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(64, activation='relu')(flat)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense1)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.4)(dense2)
    dense3 = tf.keras.layers.Dense(14, activation='relu')(dropout2)     
    dense4 = tf.keras.layers.Dense(num_actions, activation='softmax')(dense3)
    policy_network = tf.keras.Model(inputs=inputs, outputs=dense4)
    return policy_network

env = gym.make('CartPole-v1')

state_shape = env.observation_space.shape
num_actions = env.action_space.n

policy_network = build_policy_network(state_shape, num_actions)

reinforce = REINFORCE(env, policy_network, artificial_truncation=512, noise_scale=0.1)

# change some parameters of the algorithm
reinforce.gamma = 0.99
reinforce.learning_rate = 0.0001
reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, epsilon=1e-6, clipnorm=1e1)

reinforce = train(reinforce, trials=32, episodes_per_trial=8, epochs_per_trial=2, batch_size=16, verbose=True)

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



