from spurl.algorithms.reinforce import discrete
import tensorflow as tf
import numpy as np

class REINFORCE(discrete.REINFORCE):
    def __init__(self, env, policy_network, learning_rate=0.001, gamma=0.99, artificial_truncation=None):
        super().__init__(env, policy_network, learning_rate, gamma, artificial_truncation)