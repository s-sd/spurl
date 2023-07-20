import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def save_model(algorithm, save_path):
    """
    Saves trained model to specified save path
    
    Parameters:
        algorithm (REINFORCE object) 
        save_path (str) : file path for saving model 
    """
    algorithm.policy_network.save(save_path)
    print(f'Model saved to {save_path}')

def load_model(algorithm, model_path):
    """
    Loads previously trained model  
    
    Parameters:
        algorithm (REINFORCE object) 
        model_path (str) : file path of model 
        
    Returns: 
        algorithm (REINFORCE object) 
    """
    model = tf.keras.models.load_model(model_path)
    algorithm.policy_network = model
    return algorithm

def save_environment_render(rendering_env, algorithm, save_path):
    """
    Saves model environment render for visualisation 
    
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    algorithm.env = rendering_env
    state, _ = algorithm.env.reset()
    step = 0
    while True:
        action = algorithm.select_action(state)
        state, reward, done, _, _ = algorithm.env.step(np.squeeze(np.array(action, dtype=np.uint32)))
        image = algorithm.env.render()
        plt.imsave(os.path.join(save_path, f'step_{step}.png'), image) # change to save image
        step += 1
        if done:
            print(f'Trajectory saved to {save_path}')
            break

def build_cnn(state_shape, output_shape, layers):
    """
    Builds a simple CNN-based policy network with variable number of layers
    
    Parameters:
        state_shape (tuple) : shape of input state 
        output_shape (tuple) : shape of output (actions or value)
        layers (array) : layers=[[2, 3, 3],[32, 32,]] # first is list of cnn blocks, second is number of nodes 
    
    Returns:
        tf.keras.Model : Policy network model using CNNs
        
    Example:
        state_shape = (84, 84, 4)
        output_shape = 6
        layers=[[2, 3, 3],[32, 32,]]
        model = build_policy_network(state_shape, output_shape, num_actions)
    """
    
    # layers
    cnn_blocks = layers[0] 
    fcn_blocks = layers[1]
    
    model = tf.keras.models.Sequential()
    
    # Add cnn blocks 
    for i, filter_size in enumerate(cnn_blocks): 
        
        base_num = 5 #2**5 = 32 
        input_size = 2**(base_num+i) # to get 32, 64, 128 for filter size etc depending on number of blocks provided 
        
        
        if i == 0:
            model.add(tf.keras.layers.Conv2D(input_size, (filter_size, filter_size), activation='relu', input_shape=state_shape))
        else:
            model.add(tf.keras.layers.Conv2D(input_size, (filter_size, filter_size), activation='relu'))
    
    # Add maxpool and flatten layers 
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    
    # Add fcn layers 
    for j, node_size in enumerate(fcn_blocks):
        model.add(tf.keras.layers.Dense(node_size, activation='relu'))
        
    # Add final output layer
    model.add(tf.keras.layers.Dense(output_shape))
    #model.summary()
    
    return model

def build_fcn(state_shape, output_shape, layers, add_dropout = True): 
    """
    Builds a fully connected network to be used as a policy network
    
    Parameters: 
        state_shape (tuple) : shape of input state 
        output_shape (tuple) : shape of output (actions or value)
        layers (array) : layers = [32, 32] # first is list of cnn blocks, second is number of nodes 
    
    Returns:
        tf.keras.Model : Policy network model using dense layers only 
        
    Example:
        state_shape = (84, 84, 4)
        output_shape = 6
        layers=[[2, 3, 3],[32, 32,]]
        model = build_policy_network(state_shape, output_shape, num_actions)
        
    """
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=state_shape))
    model.add(tf.keras.layers.Flatten())
    
    # add fcn layers 
    for node_size in layers: 
        print(f"node size : {node_size} ")
        model.add(tf.keras.layers.Dense(node_size, activation='relu'))
        if add_dropout:
            model.add(tf.keras.layers.Dropout(0.4))
    
    # final output layer 
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    
    # print model summary
    model.summary()
    return model 

def build_policy_network(state_shape, output_shape, action_space, policy_type, layers): 
    
    """
    
    Builds a policy network using Keras
    
    Parameters:
        state_shape (tuple) : shape of input state 
        output_shape (tuple) : shape of output (actions or value)
        action_space (str) : Defines the action type as 'continuous' 'discrete' or 'multi-discrete'
        policy_type (str) : Defines type of network used (conv or fully connected)
        layers (array) : layers=[[2, 3, 3],[32, 32,]] # first is list of cnn blocks, second is number of nodes 
    
    Returns:
        tf.keras.Model : Policy network model
        
    Example:
        state_shape = (84, 84, 4)
        output_shape = 6
        layers=[[2, 3, 3],[32, 32,]]
        model = build_policy_network(state_shape, num_actions)
        
    """
    # TODO : check action space type 
    
    # initialise policy network model to be used 
    match policy_type: 
        case 'cnn':
            model = build_cnn(state_shape, output_shape, layers)
        case 'fcn':
            model = build_fcn(state_shape, output_shape, layers)
    
    return model 

if __name__ == '__main__':
     
    model_cnn = build_cnn([32,32,3], 3, layers = [[2, 3, 3],[32, 32,]]) 
    model_fcn = build_fcn([32,32,3], 3, [64,32,14])
    model_test = build_policy_network([32,32,3], 3, POLICY_TYPE = 'cnn')


    