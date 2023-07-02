import tensorflow as tf
import gymnasium as gym
import numpy as np

def train(algorithm, trials, episodes_per_trial, epochs_per_trial, batch_size, verbose):
    for trial in range(trials):
        print(f'Trial: {trial+1}/{trials}')
        states, actions, rewards = algorithm.run(episodes_per_trial)
        algorithm.train(states, actions, rewards, epochs_per_trial, batch_size, verbose=verbose)
    return algorithm

def test(algorithm, trials, episodes_per_trial):
    mean_rewards = []
    for trial in range(trials):
        print(f'Test Trial: {trial+1}/{trials}')
        states, actions, rewards = algorithm.run(episodes_per_trial)
        mean_reward = np.mean(rewards)
        mean_rewards.append(mean_reward)
        print(f'    Mean Reward: {mean_reward}')
    return mean_rewards
    

