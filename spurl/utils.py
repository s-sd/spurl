import tensorflow as tf
import gymnasium as gym
import numpy as np

def build_policy_network(state_shape, num_actions):
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    dense2 = tf.keras.layers.Dense(num_actions, activation='softmax')(dense1)
    policy_network = tf.keras.Model(inputs=inputs, outputs=dense2)
    return policy_network



class DummyEnv(gym.Env):
    def __init__(self, num_actions):
        self.episode_terminate = 256
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(32, 32))
        self.step_num = 0
    def step(self, action):
        if self.step_num >= self.episode_terminate-1:
            done = True
        else:
            done = False
        self.step_num += 1
        return np.random.rand(32, 32), np.random.rand(), done, {}
    def reset(self):
        self.step_num = 0
        return np.random.rand(32, 32)


# num_actions = 4

# env = DummyEnv()

# policy_network = build_policy_network((32,32), num_actions)


# reinforce = REINFORCE(env, policy_network)

# st, ac, re = reinforce.run(4)


# reinforce.train(st, ac, re, 2, 11)


