from spurl.algorithms.reinforce import REINFORCE
from spurl.core import train, test

import tensorflow as tf
import gymnasium as gym
import numpy as np

def build_policy_network(state_shape, num_actions):
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(256, activation='relu')(flat)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    dense3 = tf.keras.layers.Dense(num_actions, activation='softmax')(dense2)
    policy_network = tf.keras.Model(inputs=inputs, outputs=dense3)
    return policy_network

env = gym.make('CartPole-v1')

state_shape = env.observation_space.shape
num_actions = env.action_space.n

policy_network = build_policy_network(state_shape, num_actions)

reinforce = REINFORCE(env, policy_network)

# change some parameters of the algorithm
reinforce.gamma = 0.99
reinforce.learning_rate = 0.001


