from spurl.algorithms.reinforce import self_play_sequential_discrete
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class REINFORCE(self_play_sequential_discrete.REINFORCE):
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
        loss = 0
        for t in range(len(rewards)):
            state = np.array([states[t]])
            action = actions[t]
            reward = rewards[t]
            
            action_prob = self.policy_network(state)
            dist = tfp.distributions.Normal(action_prob, self.scale)  
            log_prob = dist.log_prob(action)
            loss -= log_prob * reward 
        return loss

