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
    x_train = x_train.flatten()
    x_test = x_test.flattten()

    sum = sum(x_train)
    x_train = x_train / sum
    sum = sum(x_test)
    x_test = x_test / sum

    
    return x_train, x_test, y_train, y_test, x_val, y_val

