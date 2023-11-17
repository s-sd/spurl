import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

def train(algorithm, trials, episodes_per_trial, epochs_per_trial, batch_size, verbose):
    for trial in range(trials):
        print(f'\nTrial: {trial+1}/{trials}')
        print('    Running environment')
        states, actions, rewards = algorithm.run(episodes_per_trial)
        print('    Updating policy network')
        algorithm.update(states, actions, rewards, epochs_per_trial, batch_size, verbose=verbose)
    return algorithm

def test(algorithm, trials, episodes_per_trial, deterministic=False):
    mean_rewards = []
    for trial in range(trials):
        print(f'\nTest Trial: {trial+1}/{trials}')
        states, actions, rewards = algorithm.run(episodes_per_trial, discount_rewards=False, deterministic=deterministic)
        mean_reward = np.mean(rewards)
        mean_rewards.append(mean_reward)
        print(f'    Mean Reward: {mean_reward}')
        mean_episode_length = len(rewards) / episodes_per_trial
        print(f'    Mean Episode Length: {mean_episode_length}')
    return mean_rewards, mean_episode_length

# saving was implemented within self play run algo
# def train_self_play(algorithm, trials, episodes_per_trial, epochs_per_trial, batch_size, verbose, self_play_type):
#     for trial in range(trials):
#         print(f'\nTrial: {trial+1}/{trials}')
#         print('    Running environment')
#         if trials % algorithm.opponent_save_frequency:
#             algorithm.policy_network.save(os.path.join(algorithm.opponents_path, f'{trial}'))
#         states, actions, rewards = algorithm.run(episodes_per_trial)
#         print('    Updating policy network')
#         algorithm.update(states, actions, rewards, epochs_per_trial, batch_size, verbose=verbose)
#     return algorithm

