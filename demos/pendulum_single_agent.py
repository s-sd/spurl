from spurl.algorithms.reinforce.continuous import REINFORCE
from spurl.core import train, test
from spurl.utils import save_model, load_model, save_environment_render, build_policy_network

import tensorflow as tf
import gymnasium as gym
import numpy as np
import os

tf.random.set_seed(42)
np.random.seed(42)

<<<<<<< HEAD
=======
def build_policy_network(state_shape, action_size):
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense1)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.4)(dense2)
    dense3 = tf.keras.layers.Dense(32, activation='relu')(dropout2)    
    dense4 = tf.keras.layers.Dense(np.prod(action_size), activation='tanh')(dense3)
    
    scaled_outputs = tf.keras.layers.Lambda(lambda x: (x + 1) * 2 - 2)(dense4) # scale to action space
    
    policy_network = tf.keras.Model(inputs=inputs, outputs=scaled_outputs)
    return policy_network

>>>>>>> main
env = gym.make("Pendulum-v1")

action_space = env.action_space
action_size = env.action_space.shape
state_shape = env.observation_space.shape

policy_network = build_policy_network(state_shape, 
                                      action_size = action_size, 
                                      action_space = action_space,
                                      policy_type = 'fcn',
                                      layers = [128, 64, 32],
                                      activation_fn = 'tanh')

# for linearly annealing scale
initial_scale = 4.0 # tuning this really helps training
minimum_scale = 0.2

reinforce = REINFORCE(env, policy_network, scale=initial_scale, artificial_truncation=256)

reinforce.optimizer = tf.keras.optimizers.Adam(reinforce.learning_rate, epsilon=1e-6, clipnorm=1e1)

meta_trials = 512

temp_path = r'./temp'
if not os.path.exists(temp_path):
    os.mkdir(temp_path)

for meta_trial in range(meta_trials):
    print(f'\nMeta Trial: {meta_trial+1} / {meta_trials}\n')
    reinforce = train(reinforce, trials=2, episodes_per_trial=8, epochs_per_trial=2, batch_size=16, verbose=True)
    rewards, lengths = test(reinforce, trials=1, episodes_per_trial=4, deterministic=True)
    
    # linearly annealling scale as time goes on
    reinforce.scale = initial_scale - (initial_scale - minimum_scale)*(meta_trial/meta_trials)
    
    if meta_trial % 8 == 0:
        save_model(reinforce, os.path.join(temp_path, f'model_pendulum_{meta_trial}'))

rendering_env = gym.make("Pendulum-v1", render_mode='rgb_array')

save_environment_render(rendering_env, algorithm=reinforce, save_path=os.path.join(temp_path, 'pendulum_trajectory'), deterministic=True, artificial_truncation=256)
