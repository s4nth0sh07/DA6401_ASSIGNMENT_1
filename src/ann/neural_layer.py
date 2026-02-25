"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from activations import activate

class layer:
    def __init__(self, no_of_neurons, no_of_inputs, activation = 'relu', weight_init = 'random'):
        self.no_of_inputs = no_of_inputs
        self.no_of_neurons = no_of_neurons
        if weight_init == 'random':
            self.weights = np.random.randn()
        elif weight_init == 'xavier':
            self.w
        
        self.weights = weights
        self.biases = biases
        self.activation  =activation
        self.weight_init = weight_init
    
    def forward_pass(X, W, b):
        Z = (X @ W)  + b
        return activate(Z)
    
    def backward_pass():
