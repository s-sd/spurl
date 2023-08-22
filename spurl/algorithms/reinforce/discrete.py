from spurl.algorithms.reinforce import base
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class REINFORCE(base.REINFORCE):
    def __init__(self, env, policy_network, learning_rate=0.001, gamma=0.99, noise_scale=0.1, artificial_truncation=None):
        super().__init__(env, policy_network, learning_rate, gamma, artificial_truncation)
        self.noise_scale = noise_scale
        
    def select_action(self, state, deterministic=False):
        state = np.array([state])
        action_probs = self.policy_network(state)
        
        if np.isnan(action_probs).any():
            raise ValueError(f'Network outputs contain NaN: {action_probs}') 
            # suggestions: reduce network size, clip grads, scale states, add regularisation
        
        if deterministic:
            max_prob_ind = np.argmax(np.squeeze(action_probs))
            action_probs_new = np.zeros(len(np.squeeze(action_probs)))
            action_probs_new[max_prob_ind] = 1.0
            dist = tfp.distributions.Categorical(probs=action_probs_new, dtype=tf.float32)
            action = dist.sample()
        
        else:
            noised_action_probs = np.array(action_probs) + (np.random.randn(*np.shape(action_probs)) * self.noise_scale)
            dist = tfp.distributions.Categorical(probs=noised_action_probs, dtype=tf.float32) # replcae with action_probs if no noise
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