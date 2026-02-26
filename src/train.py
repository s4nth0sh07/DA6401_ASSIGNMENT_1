"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
from utils.data_loader import load_data, process_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d', '--dataset', type = str, default='mnist', choices=['mnist', 'fashion_mnist'], help = 'Choose between mnist and fashion_mnist')
    parser.add_argument('-e', '--epochs', type = int, default=1,help = 'Number of training epochs')
    parser.add_argument('-b', '--batch_size', type = int, default=1, help = 'Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type = float, default = 0.01, help = 'Initial learning rate')
    parser.add_argument('-o', '--optimizer', type = str, default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help = 'One of sgd, momentum, nag, rmsprop, adam, nadam')
    parser.add_argument('-l', '--loss', type = str, default= 'mean_squared_error', choices = ['mean_squared_error', 'cross_entropy'], help = 'Choice of mean_squared_error or cross_entropy')
    parser.add_argument('-wd', '--weight_decay', type = float, default = 0.01, help = 'Weight decay for L2 regularization.')
    parser.add_argument('-nhl', '--num_layers', type = int, default = 1, help = 'Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type = int, default= [128], help = 'Number of neurons in each hidden layer (list of values)', nargs='+')
    parser.add_argument('-a', '--activation', type = str, default='sigmoid', choices=['sigmoid', 'tanh', 'relu'], help = 'Choice of sigmoid, tanh, relu for every hidden layer.')
    parser.add_argument('-w_i', '--weight_init', type = str, default = 'random', choices = ['random', 'xavier'], help = 'Choice of random or xavier')
    parser.add_argument('--wandb_project', type = str, default = 'ASSIGNMENT-1', help = 'W&B project name')
    parser.add_argument('--model_save_path', type = str, default='models/final_model.npy', help = 'Path to save trained model')
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    (X_train, y_train), (X_test, y_test) = load_data(args.dataset)
    X_train, X_test, y_train, y_test, X_val, y_val = process_data(X_train, X_test, y_train, y_test)

    model = NeuralNetwork(cli_args = args)
    model.train(X_train = X_train, y_train = y_train, epochs = args.epochs, batch_size = args.batch_size)

    validaton_accuracy = model.evaluate(X_val, y_val)
    testing_accuracy = model.evaluate(X_test, y_test)
    print(validaton_accuracy * 100, testing_accuracy * 100)

    print("Training complete!")


if __name__ == '__main__':
    main()
