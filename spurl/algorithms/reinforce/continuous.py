from spurl.algorithms.reinforce import base
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class REINFORCE(base.REINFORCE):
    def __init__(self, env, policy_network, scale, learning_rate=0.001, gamma=0.99, artificial_truncation=None):
        super().__init__(env, policy_network, learning_rate, gamma, artificial_truncation)
        self.scale = scale
        
    def select_action(self, state, deterministic=False):
        state = np.expand_dims(state, axis=0)
        action_probs = self.policy_network(state)
                
        if deterministic:
            dist = tfp.distributions.Normal(action_probs, 0.0)
            action = dist.sample()
        else:
            dist = tfp.distributions.Normal(action_probs, self.scale)
            action = dist.sample()
        return tf.squeeze(action).numpy()
    
    def compute_loss(self,  states, actions, rewards):
        states = np.array(states)
        action_probs = self.policy_network(states)
        dist = tfp.distributions.Normal(action_probs, self.scale) # Assume Gaussian distribution
        log_probs = dist.log_prob(actions)
        loss = -tf.reduce_mean(tf.reduce_sum(log_probs) * rewards)
        return loss