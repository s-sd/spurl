from spurl.algorithms.reinforce import REINFORCE
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class REINFORCE_Continuous(REINFORCE):
    def __init__(self, env, policy_network, scale, learning_rate=0.001, gamma=0.99, state_preprocessor=None, artificial_truncation=None):
        super().__init__(env, policy_network, learning_rate, gamma, state_preprocessor, artificial_truncation)
        self.scale = scale
        
    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.policy_network(state)
        return tf.squeeze(action).numpy()
    
    def compute_loss(self,  states, actions, rewards):
        states = np.array(states)
        logits = self.policy_network(states)
        dist = tfp.distributions.Normal(logits, self.scale) # Assume Gaussian distribution
        log_probs = dist.log_prob(actions)
        loss = -tf.reduce_mean(tf.reduce_sum(log_probs) * rewards)
        return loss

from spurl.core import train, test

import tensorflow as tf
import gymnasium as gym
import numpy as np

def build_policy_network(state_shape, action_size):
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    dense2 = tf.keras.layers.Dense(np.prod(action_size), activation='linear')(dense1)
    reshape = tf.keras.layers.Reshape(action_size)(dense2)
    policy_network = tf.keras.Model(inputs=inputs, outputs=reshape)
    return policy_network

# learn to always pick 0.5s
class DummyEnv(gym.Env):
    def __init__(self, num_actions):
        self.episode_terminate = 256
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=action_size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(32, 32))
        self.step_num = 0
    def step(self, action):
        if (action - np.array([[0.5, 0.5],[0.5, 0.5]]) < 0.2).all():
            reward = 1
        else:
            reward = 0
        
        if self.step_num >= self.episode_terminate-1:
            done = True
        else:
            done = False
        self.step_num += 1
        return np.random.rand(32, 32), reward, done, {}
    def reset(self):
        self.step_num = 0
        return np.random.rand(32, 32)

action_size = (2, 2)

env = DummyEnv(action_size)

policy_network = build_policy_network((32,32), action_size)

reinforce = REINFORCE_Continuous(env, policy_network, scale=0.2)

reinforce = train(reinforce, trials=16, episodes_per_trial=16, epochs_per_trial=2, batch_size=32, verbose=True)

history = test(reinforce, trials=2, episodes_per_trial=4)
