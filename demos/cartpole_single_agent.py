from spurl.algorithms.reinforce import REINFORCE
from spurl.core import train, test

import tensorflow as tf
import gymnasium as gym
import numpy as np

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

reinforce = REINFORCE(env, policy_network, artificial_truncation=256)

# change some parameters of the algorithm
reinforce.gamma = 0.99
reinforce.learning_rate = 0.0001
reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, epsilon=1e-6, weight_decay=0.004, clipnorm=1e1)

reinforce = train(reinforce, trials=32, episodes_per_trial=64, epochs_per_trial=2, batch_size=32, verbose=True)

rewards, lengths = test(reinforce, trials=2, episodes_per_trial=8)

# render the environment
rendering_env = gym.make('CartPole-v1', render_mode='rgb_array')
reinforce.env = rendering_env
images_list = []
state, _ = reinforce.env.reset()
while True:
    action = reinforce.select_action(state)
    state, reward, done, _, _ = reinforce.env.step(np.squeeze(np.array(action, dtype=np.uint32)))
    image = reinforce.env.render()
    images_list.append(image)
    if done:
        break


