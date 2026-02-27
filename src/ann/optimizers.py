"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
import copy

class SGD:
    def __init__(self, neural_layers, learning_rate):
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
            curr_vel_w = self.b * self.velocity_weights[name] + self.learning_rate * layer.grad_W
            curr_vel_b = self.b * self.velocity_biases[name] + self.learning_rate * layer.grad_b

            layer.weights = layer.weights - curr_vel_w
            layer.biases = layer.biases - curr_vel_b

            self.velocity_biases[name] = curr_vel_b
            self.velocity_weights[name] = curr_vel_w

class RMSprop:
    def __init__(self, neural_layers, b, e, learning_rate):
        self.layers = neural_layers
        self.learning_rate = learning_rate

        self.squared_grad_weights = {name : np.zeros_like(layer.weights) for name, layer in self.layers.items()}
        self.squared_grad_biases = {name : np.zeros_like(layer.biases) for name, layer in self.layers.items()}

        self.b, self.e = b, e

    def update(self):
        for name, layer in self.layers.items():
            curr_sg_w = self.b * self.squared_grad_weights[name] + (1 - self.b) * (layer.grad_W ** 2)
            curr_sg_b = self.b * self.squared_grad_biases[name] + (1 - self.b) * (layer.grad_b ** 2)

            temp = self.learning_rate / (np.sqrt(curr_sg_w) + self.e)
            layer.weights = layer.weights - temp * layer.grad_W
            temp = self.learning_rate / (np.sqrt(curr_sg_b) + self.e)
            layer.biases = layer.biases - temp * layer.grad_b

            self.squared_grad_biases[name] = curr_sg_b
            self.squared_grad_weights[name] = curr_sg_w

class Adam:
    def __init__(self, neural_layers, b1, b2, e, learning_rate):
        self.layers = neural_layers
        self.learning_rate = learning_rate

        self.momentum_weights = {name : np.zeros_like(layer.weights) for name, layer in self.layers.items()}
        self.velocity_weights =copy.deepcopy(self.momentum_weights)

        self.momentum_biases = {name : np.zeros_like(layer.biases) for name, layer in self.layers.items()}
        self.velocity_biases = copy.deepcopy(self.momentum_biases)
        self.b1, self.b2 = b1, b2
        self.e = e
        self.counter = 0

    def update(self):
        self.counter += 1
        for name, layer in self.layers.items():
            curr_m_w = (self.b1 * self.momentum_weights[name]) + (1 - self.b1) * layer.grad_W
            curr_v_w = (self.b2 * self.velocity_weights[name]) + (1 - self.b2) * (layer.grad_W ** 2)

            temp1 = (1 - self.b1 ** self.counter)
            temp2 = (1 - self.b2 ** self.counter)

            curr_m_w_cap = curr_m_w / temp1
            curr_v_w_cap = curr_v_w / temp2

            temp = self.learning_rate * curr_m_w_cap / (np.sqrt(curr_v_w_cap) + self.e)
            layer.weights = layer.weights - temp

            self.momentum_weights[name] = curr_m_w
            self.velocity_weights[name] = curr_v_w

            curr_m_b = (self.b1 * self.momentum_biases[name]) + (1 - self.b1) * layer.grad_b
            curr_v_b = (self.b2 * self.velocity_biases[name]) + (1 - self.b2) * (layer.grad_b ** 2)

            temp1 = (1 - self.b1 ** self.counter)
            temp2 = (1 - self.b2 ** self.counter)


            curr_m_b_cap = curr_m_b / temp1
            curr_v_b_cap = curr_v_b / temp2

            temp = self.learning_rate * curr_m_b_cap / (np.sqrt(curr_v_b_cap) + self.e)
            layer.biases = layer.biases - temp

            self.momentum_biases[name] = curr_m_b
            self.velocity_biases[name] = curr_v_b

class NAG:
    def __init__(self, neural_layers, learning_rate, b):
        self.layers = neural_layers
        self.learning_rate = learning_rate

        self.velocity_weights = {name : np.zeros_like(layer.weights) for name, layer in self.layers.items()}
        self.velocity_biases = {name : np.zeros_like(layer.biases) for name, layer in self.layers.items()}

        self.b = b
    
    def update(self):
        for name, layer in self.layers.items():
            curr_vel_w = self.b * self.velocity_weights[name] + self.learning_rate * layer.grad_W
            curr_vel_b = self.b * self.velocity_biases[name] + self.learning_rate * layer.grad_b
            layer.weights = layer.weights - (self.b * curr_vel_w + self.learning_rate * layer.grad_W)
            layer.biases = layer.biases - (self.b * curr_vel_b + self.learning_rate * layer.grad_b)

            self.velocity_weights[name] = curr_vel_w
            self.velocity_biases[name] = curr_vel_b

class Nadam:
    def __init__(self, neural_layers, b1, b2, e, learning_rate):
        self.layers = neural_layers
        self.learning_rate = learning_rate

        self.momentum_weights = {name: np.zeros_like(layer.weights) for name, layer in self.layers.items()}
        self.velocity_weights = copy.deepcopy(self.momentum_weights)
        self.momentum_biases = {name: np.zeros_like(layer.biases) for name, layer in self.layers.items()}
        self.velocity_biases = copy.deepcopy(self.momentum_biases)

        self.b1, self.b2 = b1, b2
        self.e = e
        self.counter = 0

    def update(self):
        self.counter += 1
        for name, layer in self.layers.items():
            curr_m_w = (self.b1 * self.momentum_weights[name]) + (1 - self.b1) * layer.grad_W
            curr_v_w = (self.b2 * self.velocity_weights[name]) + (1 - self.b2) * (layer.grad_W ** 2)

            curr_m_w_cap = curr_m_w / (1 - self.b1 ** self.counter)
            curr_v_w_cap = curr_v_w / (1 - self.b2 ** self.counter)
            m_w_nesterov = (self.b1 * curr_m_w_cap) + ((1 - self.b1) * layer.grad_W) / (1 - self.b1 ** self.counter)

            temp_w = self.learning_rate * m_w_nesterov / (np.sqrt(curr_v_w_cap) + self.e)
            layer.weights = layer.weights - temp_w

            self.momentum_weights[name] = curr_m_w
            self.velocity_weights[name] = curr_v_w
            curr_m_b = (self.b1 * self.momentum_biases[name]) + (1 - self.b1) * layer.grad_b
            curr_v_b = (self.b2 * self.velocity_biases[name]) + (1 - self.b2) * (layer.grad_b ** 2)

            curr_m_b_cap = curr_m_b / (1 - self.b1 ** self.counter)
            curr_v_b_cap = curr_v_b / (1 - self.b2 ** self.counter)
            m_b_nesterov = (self.b1 * curr_m_b_cap) + ((1 - self.b1) * layer.grad_b) / (1 - self.b1 ** self.counter)

            temp_b = self.learning_rate * m_b_nesterov / (np.sqrt(curr_v_b_cap) + self.e)
            layer.biases = layer.biases - temp_b
            self.momentum_biases[name] = curr_m_b
            self.velocity_biases[name] = curr_v_b
                 