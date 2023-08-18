import tensorflow as tf
import numpy as np
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
        tf.keras.utils.save_img(os.path.join(save_path, f'step_{step}.png'), image) # change to save image
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
    if action_type == 'DISCRETE':
        output = tf.keras.layers.Dense(np.prod(output_shape), activation='softmax')(input)
    elif action_type == 'CONTINUOUS':
        dense_output = tf.keras.layers.Dense(np.prod(output_shape), activation=activation_fn)(input)
        output = tf.keras.layers.Reshape(output_shape)(dense_output)
    elif action_type == 'MULTI-DISCRETE':
        raise("Not yet implemented")
    else:
        raise ValueError("Action type not recognised as discrete/continuous or multi-discrete")    
    
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
    
    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=output)
     
    return model 

def build_fcn(state_shape, output_shape, dense_layers, action_type, add_dropout = True): 
    
    """
    Builds a fully connected network to be used as a policy network
    
    Parameters: 
        state_shape (tuple) : shape of input state 
        output_shape (tuple) : shape of output (actions or value)
        layers (list) : number of nodes for each dense layer 
        action_type (str) : determines final output layer. Can take "DISCRETE", "CONTINUOUS" or "MULTIDISCRETE"
        add_dropout (bool) : Whether to include drop out layers
    Returns:
        tf.keras.Model : Policy network model using dense layers only 
        
    Example:
        state_shape = (84, 84, 4)
        output_shape = 6
        layers=[32,32] 
        action_type = 'DISCRETE'
        model = build_policy_network(state_shape, output_shape, num_actions, action_type)
    """
    
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

def build_policy_network(observation_space, action_space, policy_type, layers, activation_fn = 'linear'): 
    
    """
    Builds a policy network using Keras
    
    Parameters:
        observation_space (gymnasium.spaces object) : Defines obs space to check for input size 
        action_space (gymnasium.spaces object) : Defines action space, to check for action type and action size
        policy_type (str) : Defines type of network used (conv or fcn)
        layers (list) : For 'fcn', defines size of dense layers. For 'cnn', defines cnn filter sizes and size of dense layers
        activation_fn (str) : Defines activation function used for final output layer.
        
    Note: 
        layers : 
            - for cnn, provide a list with two separate lists, one for cnn filter size 
            and one for number of nodes for each dense layer. Ie [[2,2], [32,32]]
            - for fcn, provide a single list [] describes size of each dense layer. Ie [32,32]
        
        activation_fn : 
            - Users can define which activation to use for continuous action spaces
            - For discrete spaces, default activation function used is softmax
            
        action_space : 
            - action_type is defined internally within function, based on action_space object provided. 
            - Box spaces are defined as continuous, whilst Discrete spaces are defined as discrete action types. 
    
    Returns:
        tf.keras.Model : Policy network model
        
    Example:
    
        # For cnn use two lists : one for filter size for CNN layers, and one for number of nodes used for each dense layer
        state_shape = (84, 84, 4)
        action_size = 6
        layers=[[2, 3, 3],[32, 32]]
        model = build_policy_network(state_shape, action_size, layers)
        
        # For fcn 
        state_shape = (84, 84, 4)
        action_size = 6
        layers=[32,32]
        model = build_policy_network(state_shape, action_size, layers)
         
    """
    
    # Define state shape 
    state_shape = observation_space.shape
    
    # Checks which action space type it is: discrete, continuous or multidiscrete
    space_type = type(action_space)

    # Define action space and action size 
    if space_type == gymnasium.spaces.discrete.Discrete:
        action_type = 'DISCRETE'
        action_size = (action_space.n,)
        
    elif space_type == gymnasium.spaces.Box:
        action_type = 'CONTINUOUS'
        action_size = action_space.shape
        
    elif space_type == gymnasium.spaces.MultiDiscrete: 
        action_type = 'MULTIDISCRETE'
        action_size = (action_space.n,)
    
    else:
        raise ValueError("Action space not recognised")
    
    # initialise policy network model to be used 
    if policy_type == 'cnn': 
        model = build_cnn(state_shape, action_size, layers, action_type, activation_fn)
    elif policy_type == 'fcn':
        model = build_fcn(state_shape, action_size, layers, action_type)
    else: 
        raise ValueError("Policy type not recognised, please type 'cnn' or 'fcn' only")
    
    return model 


    