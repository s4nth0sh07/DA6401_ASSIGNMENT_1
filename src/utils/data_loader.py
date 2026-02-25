"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

def load_data(dataset):
    if dataset == 'mnist':
        return mnist.load_data()
    elif dataset == 'fashion_mnist':
        return fashion_mnist.load_data()

def process_data(x_train, x_test, y_train, y_test, val_size = 0.1):
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = val_size, random_state = 42)
    
    return x_train, x_test, y_train, y_test, x_val, y_val

