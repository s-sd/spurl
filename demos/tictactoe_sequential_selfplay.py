from spurl.algorithms.reinforce.discrete import REINFORCE
from spurl.core import train, test
from spurl.utils import save_model, load_model, save_environment_render

import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

from temp.envs.tictactoe import TicTacToeEnv

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
    dense4 = tf.keras.layers.Dense(np.prod(action_size), activation='softmax')(dense3)
    policy_network = tf.keras.Model(inputs=inputs, outputs=dense4)
    return policy_network

env = TicTacToeEnv()

state_shape = env.observation_space.shape
num_actions = env.action_space.n

policy_network = build_policy_network(state_shape, num_actions)

reinforce = REINFORCE(env, policy_network, artificial_truncation=256)

reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, epsilon=1e-6, clipnorm=1e1)

meta_trials = 64

temp_path = r'./temp'
if not os.path.exists(temp_path):
    os.mkdir(temp_path)

for meta_trial in range(meta_trials):
    print(f'\nMeta Trial: {meta_trial+1} / {meta_trials}\n')
    reinforce = train(reinforce, trials=2, episodes_per_trial=8, epochs_per_trial=2, batch_size=16, verbose=True)    
    rewards, lengths = test(reinforce, trials=1, episodes_per_trial=4, deterministic=True)
        
    # if meta_trial % 8 == 0:
        # save_model(reinforce, os.path.join(temp_path, f'model_pendulum_{meta_trial}'))

# rendering_env = gym.make("Pendulum-v1", render_mode='rgb_array')

# save_environment_render(rendering_env, algorithm=reinforce, save_path=os.path.join(temp_path, 'pendulum_trajectory'), deterministic=True, artificial_truncation=256)











