import tensorflow as tf
import numpy as np
import os

def save_model(algorithm, save_path):
    algorithm.policy_network.save(save_path)
    print(f'Model saved to {save_path}')

def load_model(algorithm, model_path):
    model = tf.keras.models.load_model(model_path)
    algorithm.policy_network = model
    return algorithm

def save_environment_render(rendering_env, algorithm, save_path, deterministic=False, artificial_truncation=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    algorithm.env = rendering_env
    state, _ = algorithm.env.reset()
    step = 0
    while True:
        action = algorithm.select_action(state, deterministic)
        reshaped_action = np.reshape(np.squeeze(np.array(action, dtype=np.uint32)), algorithm.env.action_space.shape)
        state, _, done, _, _ = algorithm.env.step(reshaped_action)
        image = algorithm.env.render()
        tf.keras.utils.save_img(os.path.join(save_path, f'step_{step}.png'), image) # change to save image
        step += 1
        if artificial_truncation is not None:
            if step > artificial_truncation:
                print(f'Trajectory saved to {save_path}')
                break
        if done:
            print(f'Trajectory saved to {save_path}')
            break
