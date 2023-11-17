from self_play_sequential_discrete import REINFORCE
from core import train, test
from wss import WSS
from wss import patchify, stitch_patches, load_detector

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import os
import shutil
import re
import warnings
import skimage


tf.random.set_seed(42)
np.random.seed(42)
image_shape = (512, 512, 3)
patch_size = (32, 32)
images = np.expand_dims(skimage.data.astronaut(), axis=0) / 255.0

def build_policy_network(state_shape, action_size):
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense1)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1);
    
    dropouta = tf.keras.layers.Dropout(0.4)(dense2)
    densea = tf.keras.layers.Dense(64, activation='relu')(dropouta);
    dropoutb = tf.keras.layers.Dropout(0.4)(densea)
    denseb = tf.keras.layers.Dense(64, activation='relu')(dropoutb);
    dropoutc = tf.keras.layers.Dropout(0.4)(denseb)
    densec = tf.keras.layers.Dense(64, activation='relu')(dropoutc);
    
    dropout2 = tf.keras.layers.Dropout(0.4)(densec)
    dense3 = tf.keras.layers.Dense(32, activation='relu')(dropout2)    
    dense4 = tf.keras.layers.Dense(np.prod(action_size[0]), activation='softmax')(dense3)
    dense5 = tf.keras.layers.Dense(np.prod(action_size[1]), activation='softmax')(dense3)
    policy_network = tf.keras.Model(inputs=inputs, outputs=[dense4, dense5])
    return policy_network

# action, termination = policy_network(np.random.randn(1, 32,32,3))

# =============================================================================
# Prioritised Fictitious Self-Play (Gaussian)
# =============================================================================


env = WSS(images, patch_size)
# obs = env.reset()

t_max = env.t_max

num_patches = (env.image_shape[0] / patch_size[0]) * (env.image_shape[1] / patch_size[1])
policy_network = build_policy_network(image_shape,
                                      action_size=[int(num_patches), 2])


class REINFORCE_WSS(REINFORCE):
    def __init__(self, env, policy_network, learning_rate=0.001, gamma=0.99, noise_scale=0.1, artificial_truncation=None, self_play_type='prioritised', opponent_save_frequency=1, opponents_path=None):
        super().__init__( env, policy_network, learning_rate, gamma, noise_scale, artificial_truncation, self_play_type, opponent_save_frequency, opponents_path)
    
    def select_action(self, policy, state, deterministic=False):
        state = np.array([state])
        action_probs, termination_probs = policy(state)
                
        if np.isnan(action_probs).any():
            raise ValueError(f'Network outputs contain NaN: {action_probs}')  
            # suggestions: reduce network size, clip grads, scale states, add regularisation
        
        # sample random actions a percentage of times
        if np.random.rand() > (1-self.noise_scale):
            patch_action = np.random.randint(np.amax(action_probs.shape))
        else:
            dist = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
            patch_action = dist.sample()
                    
        if 2 > (1-self.noise_scale):
            termination_action = np.random.randint(np.amax(termination_probs.shape))
        else:
            dist = tfp.distributions.Categorical(probs=termination_probs, dtype=tf.float32) # could add noise to action_probs
            termination_action = dist.sample()
                            
        return (int(patch_action), int(termination_action))
    
    def run(self, num_episodes, discount_rewards=True, deterministic=False):
        all_rewards = []
        all_actions = []
        all_states = []
        
        episode_number = 0
        
        for i in range(num_episodes):
            
            state = self.env.reset()
            patches = patchify(state[0], patch_size)
            
            episode_states_0 = []
            episode_actions_0 = []
            episode_rewards_0 = []
            
            episode_states_1 = []
            episode_actions_1 = []
            episode_rewards_1 = []
            
            agent_a = self.select_opponent(0)
            agent_b = self.select_opponent(1)
            
            num_opponents = len(os.listdir(self.opponents_path))
            
            if episode_number % self.opponent_save_frequency == 0 and self.self_play_type != 'vanilla':
                self.policy_network.save(os.path.join(self.opponents_path, str(num_opponents+1)))
                
            step_number = 0
            
            action_0 = self.select_action(agent_a, state[1], deterministic)
            
            patch_number_0 = int(action_0[0])
            row_0, column_0 = env.get_patch_position(patch_number_0)
            patches = env.erase_patches(patches, row_0, column_0)
            state_after_a = stitch_patches(patches, image_shape, patch_size)
            
            action_1 = self.select_action(agent_b, state_after_a, deterministic)
            
            states, rewards, dones = self.env.step(action_0, action_1)
            
            state_0, state_1 = states
            reward_0, reward_1 = rewards
            done_0, done_1 = dones
            
            episode_states_0.append(state_0)
            episode_actions_0.append(action_0)
            episode_rewards_0.append(reward_0)
            
            episode_states_1.append(state_1)
            episode_actions_1.append(action_1)
            episode_rewards_1.append(reward_1)
                
            
            while True:
                
                action_0 = self.select_action(agent_a, state_1, deterministic)
                # print(action_0)
                
                patch_number_0 = int(action_0[0])
                row_0, column_0 = env.get_patch_position(patch_number_0)
                patches = env.erase_patches(patches, row_0, column_0)
                state_after_a = stitch_patches(patches, image_shape, patch_size)
                
                action_1 = self.select_action(agent_b, state_after_a, deterministic)
                                
                states, rewards, dones = self.env.step(action_0, action_1)
                
                state_0, state_1 = states
                reward_0, reward_1 = rewards
                done_0, done_1 = dones
                
                episode_states_0.append(state_0)
                episode_actions_0.append(action_0)
                episode_rewards_0.append(reward_0)
                
                episode_states_1.append(state_1)
                episode_actions_1.append(action_1)
                episode_rewards_1.append(reward_1)
                                
                step_number += 1
                
                if self.artificial_truncation is not None:
                    if step_number > self.artificial_truncation:
                        done_1 = True
                
                if done_1:
                    episode_number += 1
                    # episode_rewards_0 = [elem[0] for elem in episode_rewards_0]
                    # episode_rewards_1 = [elem[1] for elem in episode_rewards_1]
                    if discount_rewards:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            discounted_rewards_0 = self.compute_discounted_rewards(episode_rewards_0).tolist()
                            all_rewards += discounted_rewards_0
                            discounted_rewards_1 = self.compute_discounted_rewards(episode_rewards_1).tolist()
                            all_rewards += discounted_rewards_1
                    else:
                        all_rewards += episode_rewards_0
                        all_rewards += episode_rewards_1
                                        
                    all_states += episode_states_0
                    all_actions += episode_actions_0
                    
                    all_states += episode_states_1
                    all_actions += episode_actions_1
                    
                    # print(f'Episode Reward: {np.mean(discounted_rewards)}')
                    
                    break
        
        return all_states, all_actions, all_rewards


opponents_path = r'./temp'
if os.path.exists(opponents_path):
    shutil.rmtree(opponents_path)
    os.mkdir(opponents_path)
else:
    os.mkdir(opponents_path)

reinforce = REINFORCE_WSS(env, policy_network, artificial_truncation=t_max+2, self_play_type='prioritised', opponent_save_frequency=4, opponents_path=opponents_path, noise_scale=0.05)

reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, clipnorm=1e1, weight_decay=True, epsilon=1e-5)

meta_trials = 2

for meta_trial in range(meta_trials):
    print(f'\nMeta Trial: {meta_trial+1} / {meta_trials}\n')
    reinforce = train(reinforce, trials=1, episodes_per_trial=1, epochs_per_trial=1, batch_size=4, verbose=True)    
    rewards, lengths = test(reinforce, trials=1, episodes_per_trial=4, deterministic=True)