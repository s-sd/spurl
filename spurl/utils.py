import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import gymnasium

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
        plt.imsave(os.path.join(save_path, f'step_{step}.png'), image) # change to save image
        step += 1
        if artificial_truncation is not None:
            if step > artificial_truncation:
                print(f'Trajectory saved to {save_path}')
                break
        if done:
            print(f'Trajectory saved to {save_path}')
            break

def add_final_layer(input, output_shape, action_type, activation_fn = 'linear'):
    """
    Add final layer of network, depending on action type 
    
    Parameters: 
        input ()
        output_shape (tuple)
        action_type (str)
        activation_fn (str) : Defines which activation function to use for continuous cases only
    """
    
    match action_type: 
        case 'DISCRETE':
            output = tf.keras.layers.Dense(np.prod(output_shape), activation='softmax')(input)
        case 'CONTINUOUS':
            dense_output = tf.keras.layers.Dense(np.prod(output_shape), activation=activation_fn)(input)
            output = tf.keras.layers.Reshape(output_shape)(dense_output)
        case 'MULTI-DISCRETE':
            #TODO : implement 
            raise("Not yet implemented")
        
    return output
       
def build_cnn(state_shape, output_shape, layers, action_type, add_dropout = True, activation_fn = 'linear'):
    """
    Builds a simple CNN-based policy network with variable number of layers
    
    Parameters:
        state_shape (tuple) : shape of input state 
        output_shape (tuple) : shape of output (actions or value)
        layers (array) : layers=[[2, 3, 3],[32, 32,]] # first is list of cnn blocks, second is number of nodes 
        action_space (str) : Action space which defines final layer of cnn 
        
    Returns:
        tf.keras.Model : Policy network model using CNNs
        
    Example:
        state_shape = (84, 84, 4)
        output_shape = 6
        layers=[[2, 3, 3],[32, 32,]]
        action_space = 'DISCRETE'
        model = build_policy_network(state_shape, output_shape, num_actions, 'DISCRETE')
    """
    
    # size of layers
    cnn_blocks = layers[0] 
    fcn_blocks = layers[1]
    num_fcn_layers = len(fcn_blocks)
    
    inputs = tf.keras.layers.Input(shape = state_shape)
    
    # CNN blocks  
    for i, filter_size in enumerate(cnn_blocks): 
        
        base_num = 5 #2**5 = 32 
        input_size = 2**(base_num+i) # to get 32, 64, 128 for filter size etc depending on number of blocks provided 
        
        if i == 0:
            cnn_layers = tf.keras.layers.Conv2D(input_size, (filter_size, filter_size), activation='relu', input_shape=state_shape)(inputs)
        else:
            cnn_layers = (tf.keras.layers.Conv2D(input_size, (filter_size, filter_size), activation='relu'))(cnn_layers)
    
    # Add maxpool and flatten layers 
    maxpool = tf.keras.layers.MaxPooling2D((2, 2))(cnn_layers)
    flat = tf.keras.layers.Flatten()(maxpool)
    
    # Add fcn layers 
    
    for j, node_size in enumerate(fcn_blocks):
        
        if j == 0: 
            dense = tf.keras.layers.Dense(node_size, activation='relu')(flat)
        else:
            dense = tf.keras.layers.Dense(node_size, activation='relu')(dense)
        
        # Add dropout layers for all layers, except final dense layer 
        if add_dropout and (j != (num_fcn_layers-1)): 
            dense = tf.keras.layers.Dropout(0.4)(dense) 
            
    # Final output layers 
    output = add_final_layer(dense, output_shape, action_type, activation_fn)
    
    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=output)
     
    return model 

def build_fcn(state_shape, output_shape, layers, action_type, add_dropout = True): 
    
    """
    Builds a fully connected network to be used as a policy network
    
    Parameters: 
        state_shape (tuple) : shape of input state 
        output_shape (tuple) : shape of output (actions or value)
        layers (list) : layers as a list. Leave first item blank, as only for []
        action_type (str) : determines final output layer. Can take "DISCRETE", "CONTINUOUS" or "MULTIDISCRETE"
        add_dropout (bool) : Whether to include drop out layers
    Returns:
        tf.keras.Model : Policy network model using dense layers only 
        
    Example:
        state_shape = (84, 84, 4)
        output_shape = 6
        layers=[[],[32, 32,]] # leave first item blank as only for cnn
        action_type = 'DISCRETE'
        model = build_policy_network(state_shape, output_shape, num_actions, action_type)
        
    """
    
    dense_layers = layers[1]
    num_layers = len(dense_layers) 
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    
    # Dense layers, using input nodes for each layer
    for i in range(num_layers):
        
        if i == 0: # apply previous layer for first layer 
            dense = tf.keras.layers.Dense(dense_layers[i], activation='relu')(flat)
        else:
            dense = tf.keras.layers.Dense(dense_layers[i], activation='relu')(dense)

        if add_dropout and (i != (num_layers-1)):
            dense = tf.keras.layers.Dropout(0.4)(dense) 
    
    dense_output = add_final_layer(dense, output_shape, action_type)
    model = tf.keras.Model(inputs=inputs, outputs=dense_output)
    return model 

def build_policy_network(state_shape, output_shape, action_space, policy_type, layers, activation_fn = 'linear'): 
    
    """
    Builds a policy network using Keras
    
    Parameters:
        state_shape (tuple) : shape of input state 
        output_shape (tuple) : shape of output (actions or value)
        action_space (str) : Defines the action type as 'continuous' 'discrete' or 'multi-discrete'
        policy_type (str) : Defines type of network used (conv or fully connected)
        layers (array) : layers=[[2, 3, 3],[32, 32]] # first is list of cnn blocks, second is number of nodes 
        activation_fn (str) : Defines activation function used for final output layer.
        
    Note: 
        for layers : for cnn specify cnn, fcn blocks as [[], []]
        for fcn, leave first as blank [[], [x,x]] 
        
        activation_fn : Use can define which activation to use for continuous action spaces, 
        but for discrete spaces default activation function used is softmax
    
    Returns:
        tf.keras.Model : Policy network model
        
    Example:
    
        # For cnn 
        state_shape = (84, 84, 4)
        output_shape = 6
        layers=[[2, 3, 3],[32, 32]]
        model = build_policy_network(state_shape, num_actions)
        
        # For fcn 
        layers=[[32,32]]
        
        
    """
        
    match type(action_space): 
        case gymnasium.spaces.discrete.Discrete:
            action_type = 'DISCRETE'
        case gymnasium.spaces.Box:
            action_type = 'CONTINUOUS'
        case gymnasium.spaces.MultiDiscrete:
            action_type = 'MULTIDISCRETE'
    
    # initialise policy network model to be used 
    match policy_type: 
        case 'cnn':
            model = build_cnn(state_shape, output_shape, layers, action_type, activation_fn)
        case 'fcn':
            model = build_fcn(state_shape, output_shape, layers, action_type)
    
    return model 


    