from spurl.algorithms.reinforce import REINFORCE
from spurl.core import train, test

import tensorflow as tf
import gymnasium as gym
import numpy as np

def build_policy_network(state_shape, num_actions):
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    dropout1 = tf.keras.layers.Dropout(0.5)(dense1)
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
    dense3 = tf.keras.layers.Dense(num_actions, activation='softmax')(dense2)
    policy_network = tf.keras.Model(inputs=inputs, outputs=dense3)
    return policy_network

env = gym.make('CartPole-v1')

state_shape = env.observation_space.shape
num_actions = env.action_space.n

policy_network = build_policy_network(state_shape, num_actions)

# def state_preprocessor(states):
#     for state in states

reinforce = REINFORCE(env, policy_network)

# change some parameters of the algorithm
reinforce.gamma = 0.99
reinforce.learning_rate = 0.0001

reinforce = train(reinforce, trials=16, episodes_per_trial=16, epochs_per_trial=2, batch_size=16, verbose=True)

rewards = test(reinforce, trials=2, episodes_per_trial=8)


