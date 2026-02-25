"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

def relu(x):
    return np.maximum(0, x)

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu_grad(x):
    return np.where(x > 0, 1.0, 0.0)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - (tanh(x) ** 2)

def softmax(x):
    # res = np.zeros(x)

    # for i in range(x.shape[0]):
    #     curr = x[i]
    #     curr_max = np.max(curr)
    #     temp = np.exp(curr - curr_max)
    #     curr_sum = np.sum(temp)
    #     res[i] = temp / curr_sum
    
    # return res
    maxi = np.max(x, axis = 1, keepdims = True)
    temp = np.exp(x - maxi)
    summer = np.sum(temp, axis = 1, keepdims = True)

    return temp / summer

def forward_activate(activation):
    match activation.lower():
        case 'relu':
            return relu
        case 'sigmoid':
            return sigmoid
        case 'tanh':
            return tanh
        case 'softmax':
            return softmax

def backward_activate(activation):
    match activation.lower():
        case 'relu':
            return relu_grad
        case 'sigmoid':
            return sigmoid_grad
        case 'tanh':
            return tanh_grad
        