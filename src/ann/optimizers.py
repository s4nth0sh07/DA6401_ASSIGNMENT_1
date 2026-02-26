"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class SGD:
    def __init__(self, neural_layers, learning_rate)
        self.layers = neural_layers
        self.learning_rate = learning_rate
    
    def update(self):
        for name, layer in self.layers.items():
            layer.weights = layer.weights - (self.learning_rate * layer.grad_W)
            layer.biases = layer.biases - (self.learning_rate * layer.grad_b)

class Momentum:
    def __init__(self, neural_layers, learning_rate, b):
        self.layers = neural_layers
        self.learning_rate = learning_rate

        self.velocity_weights = {name : np.zeros_like(layer.weights) for name, layer in self.layers.items()}
        self.velocity_biases = {name : np.zeros_like(layer.biases) for name, layer in self.layers.items()}

        self.b = b
    
    def update(self):
        for name, layer in self.layers.items():
            curr_vel_w = self.b * self.v_w[name] + self.learning_rate * layer.grad_W
            curr_vel_b = self.b * self.v_b[name] + self.learning_rate * layer.grad_b

            layer.weights = layer.weights - curr_vel_w
            layer.biases = layer.biases - curr_vel_b

class RMSprop:
    def __init__(self, neural_layers, b, e, learning_rate):
        self.layers = neural_layers
        self.learning_rate = learning_rate

        self.squared_grad_weights = {name : np.zeros_like(layer.weights) for name, layer in self.layers.items()}
        self.squared_grad_biases = {name : np.zeros_like(layer.biases) for name, layer in self.layers.items()}

        self.b, self.e = b, e

    def update(self):
        for name, layer in self.layers.items():
            curr_sg_w = self.b * self.v_w[name] + (1 - self.b) * (layer.grad_w ** 2)
            curr_sg_b = self.b * self.v_b[name] + (1 - self.b) * (layer.grad_b ** 2)

            temp = (self.learning_rate / np.sqrt(curr_sg_w) + self.e)
            layer.weights = layer.weights - temp * layer.grad_W
            temp = (self.learning_rate / np.sqrt(curr_sg_b) + self.e)
            layer.biases = layer.biases - temp * layer.grad_b

            self.squared_grad_biases[name] = curr_sg_b
            self.squared_grad_weights[name] = curr_sg_w

class Adam:
    def __init__(self, neural_layers, b1, b2, epsilon, learning_rate):
        self.layers = neural_layers
        self.learning_rate = learning_rate

        self.
        self.b1, self.b2 = b1, b2
        self.epsilon = epsilon
        self.counter = 0