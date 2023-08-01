from spurl.algorithms.reinforce import base
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class REINFORCE(base.REINFORCE):
    def __init__(self, env, policy_network, learning_rate=0.001, gamma=0.99, artificial_truncation=None):
        super().__init__(env, policy_network, learning_rate, gamma, artificial_truncation)
        
    def select_action(self, state, deterministic=False):
        state = np.array([state])
        action_probs = self.policy_network(state)
        
        if np.isnan(action_probs).any():
            raise ValueError('Network outputs contains NaN') 
            # suggestions: reduce network size, clip grads, scale states, add regularisation
        
        if deterministic:
            dist = tfp.distributions.Categorical(probs=(action_probs), dtype=tf.float32)
            action = dist.sample()
        
        else:
            dist = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
            action = dist.sample()
        return action
    
    def compute_loss(self, states, actions, rewards):
        loss = 0
        for t in range(len(rewards)):
            state = np.array([states[t]])
            action = actions[t]
            reward = rewards[t]
            
            action_probs = self.policy_network(state)
            log_prob = tf.math.log(action_probs[0, int(action)])
            loss -= log_prob * reward
        return loss