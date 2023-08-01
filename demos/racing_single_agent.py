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
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense1)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.4)(dense2)
    dense3 = tf.keras.layers.Dense(32, activation='relu')(dropout2)    
    dense4 = tf.keras.layers.Dense(np.prod(action_size), activation='linear')(dense3)
    policy_network = tf.keras.Model(inputs=inputs, outputs=dense4)
    return policy_network

env = gym.make("CarRacing-v2")

action_size = env.action_space.shape
state_shape = env.observation_space.shape

policy_network = build_policy_network(state_shape, action_size)

reinforce = REINFORCE(env, policy_network, scale=0.4, artificial_truncation=256)

reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, epsilon=1e-6, clipnorm=1e1)

all_rewards = []
all_lengths = []

for i in range(4):
    reinforce = train(reinforce, trials=2, episodes_per_trial=16, epochs_per_trial=8, batch_size=16, verbose=True)    
    rewards, lengths = test(reinforce, trials=1, episodes_per_trial=4, deterministic=True)

    all_rewards.append(rewards)
    all_lengths.append(lengths)

temp_path = r'./temp'
if not os.path.exists(temp_path):
    os.mkdir(temp_path)

rendering_env = gym.make("CarRacing-v2", render_mode='rgb_array')

save_environment_render(rendering_env, algorithm=reinforce, save_path=os.path.join(temp_path, 'racing_trajectory'), deterministic=False, artificial_truncation=2048)



state, _ = env.reset()
reinforce.select_action(state)
