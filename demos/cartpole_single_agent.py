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

initial_observation, _ = env.reset()

np.shape(initial_observation)

a,b,c,d,e = env.step(env.action_space.sample())
