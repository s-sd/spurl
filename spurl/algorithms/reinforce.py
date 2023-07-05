import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

class REINFORCE:
    def __init__(self, env, policy_network, learning_rate=0.001, gamma=0.99, state_preprocessor=None, artificial_truncation=None):
        
        self.env = env
        
        self.num_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy_network = policy_network
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        self.rewards = []
        self.actions = []
        self.states = []
        
        self.state_preprocessor = state_preprocessor
        
        self.artificial_truncation = artificial_truncation
    
    def select_action(self, state):
        state = np.array([state])
        action_probs = self.policy_network(state)
        if np.isnan(action_probs).any():
            print(action_probs)
            print(state)
            raise ValueError('Network outputs contains NaN') 
            # suggestions: reduce network size, clip grads, scale states, add regularisation
        
        dist = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
        action = dist.sample()
        return action
    
    def compute_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros(len(rewards))
        running_reward = 0
        for t in reversed(range(len(rewards))):
            running_reward = rewards[t] + self.gamma * running_reward
            discounted_rewards[t] = running_reward
        normalised_discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8) # normalise
        return normalised_discounted_rewards
    
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
    
    def run(self, num_episodes, discount_rewards=True):
        rewards = []
        actions = []
        states = []
        for i in range(num_episodes):
            state = self.env.reset()
            
            if type(state) is tuple and len(state) > 1:
                state, _ = state
            
            episode_rewards = []
            episode_actions = []
            episode_states = []
            
            episode_number = 0
            
            while True:
                action = self.select_action(state)
                values = self.env.step(np.squeeze(np.array(action, dtype=np.uint32)))
                
                if type(values) is tuple and len(values)>4:
                    next_state, reward, done, _, _ = values
                else:
                    next_state, reward, done, _ = values
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                state = next_state
                
                #terminate episode at artificial truncation number of steps
                if self.artificial_truncation is not None:
                    if episode_number > self.artificial_truncation:
                        done = True
                
                episode_number += 1
                
                if done:
                    if discount_rewards:
                        discounted_rewards = self.compute_discounted_rewards(episode_rewards).tolist()
                        rewards += discounted_rewards
                    else:
                        rewards += episode_rewards
                                        
                    states += episode_states
                    actions += episode_actions
                    
                    # print(f'Episode Reward: {np.mean(discounted_rewards)}')
                    
                    break
        
        if self.state_preprocessor is not None:
            states = self.state_preprocessor(states)
        
        return states, actions, rewards
    
    def update(self, states, actions, rewards, epochs, batch_size, verbose=True):
        for epoch in range(epochs):
            if verbose:
                print(f'Epoch: {epoch+1}/{epochs}')
            for start_index in tqdm(range(0, len(states), batch_size), disable = not verbose):
                states_batch = states[start_index:start_index+batch_size]
                actions_batch = actions[start_index:start_index+batch_size]
                rewards_batch = rewards[start_index:start_index+batch_size]
                with tf.GradientTape() as tape:
                    loss = self.compute_loss(states_batch, actions_batch, rewards_batch)        
                self.optimizer.minimize(loss, self.policy_network.trainable_variables, tape)