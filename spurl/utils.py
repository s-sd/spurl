import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import gym

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

def build_cnn_refactor(state_shape, output_shape, layers, action_type):
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
            
    # Final output layers 
    output = add_final_layer(dense, output_shape, action_type)
    
    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=output)
     
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
    #model.summary()
    return model 

def add_final_layer(input, output_shape, action_type):
    """
    Add final layer of network, depending on action type 
    """
    match action_type: 
        case 'DISCRETE':
            dense_output = tf.keras.layers.Dense(output_shape, activation='softmax')(input)
        case 'CONTINUOUS':
            dense_output = tf.keras.layers.Dense(output_shape, activation='linear')(input)
        case 'MULTI-DISCRETE':
            #TODO : implement 
            raise("Not yet implemented")
        
    return dense_output 
        
def build_fcn_refactor(state_shape, output_shape, layers, action_type, add_dropout = True): 
    
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
    
    num_layers = len(layers) 
    inputs = tf.keras.layers.Input(shape=state_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    
    # Dense layers, using input nodes for each layer
    for i in range(num_layers):
        if i == 0: # apply previous layer for first layer 
            dense = tf.keras.layers.Dense(layers[i], activation='relu')(flat)
        else:
            dense = tf.keras.layers.Dense(layers[i], activation='relu')(dense)
        
        # Add drop out if needed
        if add_dropout: 
            dense = tf.keras.layers.Dropout(0.4)(dense) 
    
    dense_output = add_final_layer(dense, 3, action_type)
    model = tf.keras.Model(inputs=inputs, outputs=dense_output)
    return model 

def build_policy_network(state_shape, output_shape, action_space, policy_type, layers): 
    
    """
    Builds a policy network using Keras
    
    Parameters:
        state_shape (tuple) : shape of input state 
        output_shape (tuple) : shape of output (actions or value)
        action_space (str) : Defines the action type as 'continuous' 'discrete' or 'multi-discrete'
        policy_type (str) : Defines type of network used (conv or fully connected)
        layers (array) : layers=[[2, 3, 3],[32, 32]] # first is list of cnn blocks, second is number of nodes 
        
    Note: 
        for layers : for cnn specify cnn, fcn blocks as [[], []]
        for fcn, simple [] will suffice!!! 
    
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
        case gym.spaces.Discrete:
            action_type = 'DISCRETE'
        case gym.spaces.Box:
            action_type = 'CONTINUOUS'
        case gym.spaces.MultiDiscrete:
            action_type = 'MULTIDISCRETE'
    print(f"Action type: {action_type}")
    
    # initialise policy network model to be used 
    match policy_type: 
        case 'cnn':
            model = build_cnn(state_shape, output_shape, layers)
        case 'fcn':
            model = build_fcn(state_shape, output_shape, layers)
    
    return model 

if __name__ == '__main__':
     
    env = gym.make('CartPole-v1')

    state_shape = env.observation_space.shape
    action_space = env.action_space
    num_actions = env.action_space.n
    
    model_cnn = build_cnn([32,32,3], 3, layers = [[2, 3, 3],[32, 32,]]) 
    model_cnn2 = build_cnn_refactor([32,32,3], 3, layers = [[2, 3, 3],[32, 32,]], action_type = 'DISCRETE')
    
    #model_fcn1 = build_fcn([32,32,3], 3, [64,32,14])
    #model_fcn2 = build_fcn_refactor([32,32,3], 2, [64,32,14], action_type = 'DISCRETE')
    
    #model_test = build_policy_network([32,32,3], 3, action_space = action_space, policy_type = 'fcn', layers = [32, 32])
    
    print('chicken')
    