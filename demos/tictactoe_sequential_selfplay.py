from spurl.algorithms.reinforce.discrete import REINFORCE
from spurl.core import train, test
from spurl.utils import save_model, load_model, save_environment_render, build_policy_network

import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

from temp.envs.tictactoe import TicTacToeEnv

tf.random.set_seed(42)
np.random.seed(42)

env = TicTacToeEnv()

state_shape = env.observation_space.shape
action_space = env.action_space
num_actions = (action_space.n,)

policy_network = build_policy_network(state_shape,
                                      action_size = num_actions,
                                      action_space = action_space,
                                      policy_type = 'fcn',
                                      layers = [128, 64, 32])

reinforce = REINFORCE(env, policy_network, artificial_truncation=256)

reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, epsilon=1e-6, clipnorm=1e1)

meta_trials = 64

temp_path = r'./temp'
if not os.path.exists(temp_path):
    os.mkdir(temp_path)

for meta_trial in range(meta_trials):
    print(f'\nMeta Trial: {meta_trial+1} / {meta_trials}\n')
    reinforce = train(reinforce, trials=2, episodes_per_trial=8, epochs_per_trial=2, batch_size=16, verbose=True)    
    rewards, lengths = test(reinforce, trials=1, episodes_per_trial=4, deterministic=True)
        
    # if meta_trial % 8 == 0:
        # save_model(reinforce, os.path.join(temp_path, f'model_pendulum_{meta_trial}'))

# rendering_env = gym.make("Pendulum-v1", render_mode='rgb_array')

# save_environment_render(rendering_env, algorithm=reinforce, save_path=os.path.join(temp_path, 'pendulum_trajectory'), deterministic=True, artificial_truncation=256)












