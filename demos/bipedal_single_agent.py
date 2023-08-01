from spurl.algorithms.reinforce.continuous import REINFORCE
from spurl.core import train, test
from spurl.utils import save_model, load_model, save_environment_render

import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

tf.random.set_seed(42)
np.random.seed(42)

def build_policy_network(state_shape, action_size):
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(64, activation='relu')(flat)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense1)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.4)(dense2)
    dense3 = tf.keras.layers.Dense(24, activation='relu')(dropout2)    
    dense4 = tf.keras.layers.Dense(np.prod(action_size), activation='linear')(dense3)
    policy_network = tf.keras.Model(inputs=inputs, outputs=dense4)
    return policy_network

env = gym.make("BipedalWalker-v3", hardcore=False)

action_size = env.action_space.shape
state_shape = env.observation_space.shape

policy_network = build_policy_network(state_shape, action_size)

# for linearly annealing scale
initial_scale = 0.8
minimum_scale = 0.2

reinforce = REINFORCE(env, policy_network, scale=initial_scale)

reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, epsilon=1e-6, clipnorm=1e1)

all_rewards = []
all_lengths = []

total_trials = 64

for trial in range(total_trials):
    reinforce = train(reinforce, trials=4, episodes_per_trial=8, epochs_per_trial=2, batch_size=32, verbose=True)
    rewards, lengths = test(reinforce, trials=2, episodes_per_trial=4, deterministic=True)
    
    # linearly annealling scale as time goes on
    reinforce.scale = initial_scale - (initial_scale - minimum_scale)*(trial/total_trials)
    
    all_rewards.append(rewards)
    all_lengths.append(lengths)
    

temp_path = r'./temp'
if not os.path.exists(temp_path):
    os.mkdir(temp_path)

save_model(reinforce, os.path.join(temp_path, 'model_bipedal'))

rendering_env = gym.make("BipedalWalker-v3", hardcore=False, render_mode='rgb_array')

save_environment_render(rendering_env, algorithm=reinforce, save_path=os.path.join(temp_path, 'bipedal_trajectory'), deterministic=True)
