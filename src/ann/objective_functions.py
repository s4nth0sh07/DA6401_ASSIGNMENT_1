"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    temp = y_true * np.log(y_pred + 1e-9)
    cel = -1 * np.sum(temp)
    return cel/y_true.shape[0]

def cross_entropy_grad(y_true, y_pred):
    return y_pred - y_true

def mean_squared_loss(y_true, y_pred):
    temp = (y_pred - y_true) ** 2
    mse = np.sum(temp)
    return mse/y_true.shape[0]

def mean_squared_grad(y_true, y_pred):
    mse_grad = 2 * (y_pred - y_true) * (y_pred * (1 - y_pred))
    return mse_grad