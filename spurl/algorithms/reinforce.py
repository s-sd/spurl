import tensorflow as tf
import numpy as np
import gymnasium as gym 
from tqdm import tqdm

class REINFORCE:
    def __init__(self, env, policy_network, learning_rate=0.001, gamma=0.99):
        
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
    
    def select_action(self, state):
        state = np.array([state])
        action_probs = self.policy_network(state)[0]
        action = np.random.choice(self.num_actions, p=action_probs.numpy())
        return action
    
    def compute_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros(len(rewards))
        running_reward = 0
        for t in reversed(range(len(rewards))):
            running_reward = rewards[t] + self.gamma * running_reward
            discounted_rewards[t] = running_reward
        normalised_discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards) # normalise
        return normalised_discounted_rewards
    
    def compute_loss(self, states, actions, rewards):
        loss = 0
        for t in range(len(rewards)):
            state = np.array([states[t]])
            action = actions[t]
            reward = rewards[t]
            
            action_probs = self.policy_network(state)
            log_prob = tf.math.log(action_probs[0, action])
            loss -= log_prob * reward
        return loss
    
    def run(self, num_episodes):
        rewards = []
        actions = []
        states = []
        for i in range(num_episodes):
            state = self.env.reset()
            
            episode_rewards = []
            episode_actions = []
            episode_states = []
            
            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                state = next_state
                
                if done:
                    discounted_rewards = self.compute_discounted_rewards(episode_rewards).tolist()
                                        
                    states += episode_states
                    actions += episode_actions
                    rewards += discounted_rewards
                    
                    print(f'Episode Reward: {np.mean(discounted_rewards)}')
                    
                    break
        
        return states, actions, rewards
    
    def train(self, states, actions, rewards, epochs, batch_size, verbose=True):
        for epoch in range(epochs):
            print(f'Epoch: {epoch+1}/{epochs}')
            for start_index in tqdm(range(0, len(states), batch_size), disable = not verbose):
                states_batch = states[start_index:start_index+batch_size]
                actions_batch = actions[start_index:start_index+batch_size]
                rewards_batch = rewards[start_index:start_index+batch_size]
                with tf.GradientTape() as tape:
                    loss = self.compute_loss(states_batch, actions_batch, rewards_batch)        
                self.optimizer.minimize(loss, self.policy_network.trainable_variables, tape)





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


























