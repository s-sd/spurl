from spurl.algorithms.reinforce import discrete
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os

class REINFORCE(discrete.REINFORCE):
    def __init__(self, env, policy_network, learning_rate=0.001, gamma=0.99, artificial_truncation=None, self_play_type='vanilla', opponent_save_frequency=1, opponents_path=None):
        super().__init__(env, policy_network, learning_rate, gamma, artificial_truncation)
        self.self_play_type = self_play_type
        self.opponents_path = opponents_path
    
    def select_action(self, policy, state, deterministic=False):
        state = np.array([state])
        action_probs = policy(state)
        
        if np.isnan(action_probs).any():
            raise ValueError('Network outputs contains NaN') 
            # suggestions: reduce network size, clip grads, scale states, add regularisation
        
        if deterministic:
            action_probs = tf.math.round(action_probs)
            dist = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
            action = dist.sample()
    
        else:
            dist = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
            action = dist.sample()
        return action

    def opponent_sampler(self, opponents_list):
        
        dist = tfp.distributions.Normal(len(opponents_list)/2.0, len(opponents_list)/8.0)
        selected_opponent = int(np.clip(dist.sample(), 0, len(opponents_list)))
        
        return selected_opponent
    
    def select_action_self_play(self, player_num, state, deterministic=False):
        
        if player_num == 0:
            action = self.select_action(self.policy_network, state, deterministic)
        
        else:
            
            if self.self_play_type == 'vanilla':
                policy_network = self.policy_network
                
            elif self.self_play_type == 'fictitious':
                opponents_list = sorted(os.listdir(self.opponents_path))
                opponent_probs = np.ones((len(opponents_list)))
                dist = tfp.distributions.Categorical(probs=opponent_probs, dtype=tf.float32)
                selected_opponent = int(dist.sample())
                policy_network = tf.keras.models.load_model(os.path.join(self.opponents_path, opponents_list[int(selected_opponent)]))
            
            elif self.self_play_type == 'prioritised':
                opponents_list = sorted(os.listdir(self.opponents_path))
                selected_opponent = self.opponent_sampler(opponents_list)
                policy_network = tf.keras.models.load_model(os.path.join(self.opponents_path, opponents_list[int(selected_opponent)]))
                                
            action = self.select_action(policy_network, state, deterministic)
        # policy_net = tf.keras.models.load_model(opponent_path)
        # finish this function by adding action selection using opponent using the probability sampling
        return action
    
    def invert_state(self, state):
        state[:, :, 0] *= -1
        return state
        
    def run(self, num_episodes, discount_rewards=True, deterministic=False):
        # finish this function by saving models periodically
        rewards = []
        actions = []
        states = []
        
        for i in range(num_episodes):
            state = self.env.reset()
            
            if type(state) is tuple and len(state) > 1:
                state, _ = state
            
            episode_rewards_p1 = []
            episode_actions_p1 = []
            episode_states_p1 = []
            
            episode_rewards_p2 = []
            episode_actions_p2 = []
            episode_states_p2 = []
            
            episode_number = 0
            
            while True:
                
                if self.env.current_player_num == 1:
                    state = self.invert_state(state)
                    
                action = self.select_action_self_play(self.env.current_player_num, state, deterministic)
            
                reshaped_action = np.reshape(np.squeeze(np.array(action, dtype=np.uint32)), self.env.action_space.shape)
                values = self.env.step(reshaped_action)
                
                if type(values) is tuple and len(values)>4:
                    next_state, reward, done, _, _ = values
                else:
                    next_state, reward, done, _ = values
                
                if self.env.current_player_num == 0:
                    episode_states_p1.append(state)
                    episode_actions_p1.append(action)
                    episode_rewards_p1.append(reward)
                else:
                    episode_states_p2.append(state)
                    episode_actions_p2.append(action)
                    episode_rewards_p2.append(reward)
                
                state = next_state
                
                #terminate episode at artificial truncation number of steps
                if self.artificial_truncation is not None:
                    if episode_number > self.artificial_truncation:
                        done = True
                
                episode_number += 1
                
                if done:
                    episode_rewards_p1 = [elem[0] for elem in episode_rewards_p1]
                    episode_rewards_p2 = [elem[1] for elem in episode_rewards_p2]
                    if discount_rewards:
                        discounted_rewards_p1 = self.compute_discounted_rewards(episode_rewards_p1).tolist()
                        rewards += discounted_rewards_p1
                        discounted_rewards_p2 = self.compute_discounted_rewards(episode_rewards_p2).tolist()
                        rewards += discounted_rewards_p2
                    else:
                        rewards += episode_rewards_p1
                        rewards += episode_rewards_p2
                                        
                    states += episode_states_p1
                    actions += episode_actions_p1
                    
                    states += episode_states_p2
                    actions += episode_actions_p2
                    
                    # print(f'Episode Reward: {np.mean(discounted_rewards)}')
                    
                    break
        
        return states, actions, rewards