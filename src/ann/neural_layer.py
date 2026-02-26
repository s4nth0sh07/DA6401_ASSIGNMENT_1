"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from .activations import forward_activate, backward_activate

class layer:
    def __init__(self, no_of_neurons, no_of_inputs, weight_init = 'random', activation = 'relu'):
        self.no_of_inputs = no_of_inputs
        self.no_of_neurons = no_of_neurons

        if weight_init == 'random':
            self.weights = np.random.randn(no_of_inputs, no_of_neurons)
            self.weights *= 0.01
        elif weight_init == 'xavier':
            self.weights = np.random.randn(no_of_inputs, no_of_neurons)
            self.weights *= np.sqrt(2.0 / (no_of_inputs + no_of_neurons))
        
        self.biases = np.zeros((1, no_of_neurons))
        self.weight_init = weight_init
        self.forward_activation = forward_activate(activation)
        self.backward_activation = backward_activate(activation)

        self.x, self.z = None, None

        self.grad_W = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.biases)
    
    def forward_pass(self, x):
        self.x = x
        z = x @ self.weights + self.biases
        self.z = z
        res = self.forward_activation(z)
        return res
    
    def backward_pass(self, loss_grad):
        grad_z = self.backward_activation(self.z)
        delta = loss_grad * grad_z
        no_of_images = self.x.shape[0]
        self.grad_W = (np.transpose(self.x) @ delta)
        self.grad_W /= no_of_images
        self.grad_b = np.sum(delta, axis = 0, keepdims = True)
        self.grad_b /= no_of_images
        res = delta @ np.transpose(self.weights)
        return res