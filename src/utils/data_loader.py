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

def process_data(X_train, X_test, y_train, y_test, val_size = 0.1):
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    temp = np.zeros((y_train.shape[0], 10))
    temp[np.arange(y_train.shape[0]), y_train] = 1
    y_train = temp

    temp = np.zeros((y_test.shape[0], 10))
    temp[np.arange(y_test.shape[0]), y_test] = 1
    y_test = temp
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size, random_state = 42)
    
    return X_train, X_test, y_train, y_test, X_val, y_val

