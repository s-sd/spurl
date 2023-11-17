import gymnasium
import numpy as np
import skimage
import tensorflow.image as image_processing
import matplotlib.pyplot as plt
import tensorflow as tf


def patchify(image, patch_size):
    patches = image_processing.extract_patches(np.expand_dims(image, axis=0),
                                     [1, patch_size[0], patch_size[1], 1],
                                     strides=[1, patch_size[0], patch_size[1], 1],
                                     rates=[1,1,1,1],
                                     padding='SAME')
    patches = np.array(patches)
    patches_reshaped = np.squeeze(patches.reshape(patches.shape[:-1] + (patch_size[0], patch_size[1], 3)))
    return patches_reshaped

def stitch_patches(patches, image_shape, patch_size):
    columns = []
    for i in range(patches.shape[0]):
        column = np.vstack(patches[:, i])
        columns.append(column)
    image = np.hstack(columns)
    return image

# image = skimage.data.astronaut()
# patch_size = (32, 32)
# image_shape = image.shape

# patches = patchify(image, patch_size)
# stitched_image = stitch_patches(patches, image_shape, patch_size)

def build_detector(image_shape):
    inputs = tf.keras.layers.Input(shape=image_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense1)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1);
    dropout2 = tf.keras.layers.Dropout(0.4)(dense2)
    dense3 = tf.keras.layers.Dense(32, activation='relu')(dropout2)    
    dense4 = tf.keras.layers.Dense(np.prod(2), activation='softmax')(dense3)
    detector = tf.keras.Model(inputs=inputs, outputs=[dense4])
    return detector

def load_detector(image_shape):
    # put your detector training or weight loading code here
    detector = build_detector(image_shape)
    return detector

def get_detector_signal(patch_0, patch_1, detector, image_shape):
    interpolated_patch_0 = skimage.transform.resize(patch_0, image_shape)
    score_0 = np.array(detector(np.expand_dims(interpolated_patch_0, axis=0)))[-1]
    
    interpolated_patch_1 = skimage.transform.resize(patch_1, image_shape)
    score_1 = np.array(detector(np.expand_dims(interpolated_patch_1, axis=0)))[-1]
    
    winning_player = np.argmax([score_0, score_1])
    return winning_player


class WSS(gymnasium.Env):
    def __init__(self, images, patch_size):
        self.images = images
        self.patch_size = patch_size
        self.image_shape = np.shape(images)[1:]
        self.patch_shape = patch_size + (3,)
        self.detector = load_detector(self.image_shape)
        self.t_min = 2
        self.t_max = 2048
    
    def get_patch_position(self, patch_number):
        max_column = self.image_shape[1] / self.patch_size[1]
        column = patch_number % max_column
        row = np.floor(patch_number / max_column)
        return int(row), int(column)
    
    def erase_patches(self, patches, row, column):
        patches[row, column] = np.zeros(patches.shape[2:])
        return patches
    
    def step(self, action_0, action_1):
        
        patch_number_0 = int(action_0[0])
        termination_0 = int(action_0[1])
        row_0, column_0 = self.get_patch_position(patch_number_0)
        
        patch_number_1 = int(action_1[0])
        termination_1 = int(action_1[1])
        row_1, column_1 = self.get_patch_position(patch_number_1)
        
        # print(termination_0, termination_1)
        
        patch_0 = self.patches[row_0, column_0]
        patch_1 = self.patches[row_1, column_1]
        
        winning_player = get_detector_signal(patch_0, patch_1, self.detector, self.image_shape)
        
        if winning_player == 0:
            reward_0 = +1
            reward_1 = -1
        else:
            reward_0 = -1
            reward_1 = +1
            
        if termination_0 or termination_1:
            reward_0 *= 100
            reward_1 *= 100
            done = True
            if self.step_number < self.t_min or self.step_number > self.t_max:
                reward_0 += -1
                reward_0 += -1
        else:
            done = False
            
        if np.amax(patch_0) == 0:
            reward_0 += -1
        if np.amax(patch_1) == 0:
            reward_1 += -1
        
        self.patches = self.erase_patches(self.patches, row_0, column_0)
        self.patches = self.erase_patches(self.patches, row_1, column_1)
        
        self.image = stitch_patches(self.patches, self.image_shape, self.patch_size)
                    
        self.step_number += 1
         
        return (self.image, self.image), (reward_0, reward_1), (done, done)
        
    def reset(self):
        image_index = np.random.randint(0, len(self.images))
        self.image = self.images[image_index]
        self.patches = patchify(self.image, self.patch_size)
        self.step_number = 0
        return (self.image, self.image)

    
# images = np.expand_dims(skimage.data.astronaut(), axis=0)
# env = WSS(images, (32, 32))

# obs = env.reset()

# obs, rew, done = env.step([5, 0], [89,0])

# plt.imshow(obs[0])