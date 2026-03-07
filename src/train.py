"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import numpy as np
import argparse, wandb
import os
import json
from sklearn.metrics import f1_score
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
    parser.add_argument('-o', '--optimizer', type = str, default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop'], help = 'One of sgd, momentum, nag, rmsprop')
    parser.add_argument('-l', '--loss', type = str, default= 'mean_squared_error', choices = ['mean_squared_error', 'cross_entropy'], help = 'Choice of mean_squared_error or cross_entropy')
    parser.add_argument('-wd', '--weight_decay', type = float, default = 0.01, help = 'Weight decay for L2 regularization.')
    parser.add_argument('-nhl', '--num_layers', type = int, default = 1, help = 'Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type = str, default= ["128"], help = 'Number of neurons in each hidden layer (list of values)', nargs='+')
    parser.add_argument('-a', '--activation', type = str, default='sigmoid', choices=['sigmoid', 'tanh', 'relu'], help = 'Choice of sigmoid, tanh, relu for every hidden layer.')
    parser.add_argument('-w_i', '--weight_init', type = str, default = 'random', choices = ['random', 'xavier'], help = 'Choice of random or xavier')
    parser.add_argument('--wandb_project', type = str, default = 'ASSIGNMENT-1', help = 'W&B project name')
    parser.add_argument('--model_save_path', type = str, default='src/best_model.npy', help = 'Path to save trained model')
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    raw_sz = " ".join(args.hidden_size) if isinstance(args.hidden_size, list) else args.hidden_size
    args.hidden_size = [int(x.strip('[], ')) for x in raw_sz.replace(',', ' ').split()]

    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    (X_train, y_train), (X_test, y_test) = load_data(args.dataset)
    X_train, X_test, y_train, y_test, X_val, y_val = process_data(X_train, X_test, y_train, y_test)

    model = NeuralNetwork(cli_args = args)
    model.X_val = X_val
    model.y_val = y_val

    model.train(X_train=X_train, y_train=y_train, epochs=args.epochs, batch_size=args.batch_size)

    validaton_accuracy = model.evaluate(X_val, y_val)
    testing_accuracy = model.evaluate(X_test, y_test)
    
    logits = model.forward(X_test)
    y_true_idx = np.argmax(y_test, axis=1)
    y_pred_idx = np.argmax(logits, axis=1)
    test_f1 = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)

    print(validaton_accuracy * 100, testing_accuracy * 100)
    
    wandb.log({"test_accuracy": testing_accuracy, "test_f1": test_f1})

    best_weights = model.get_weights()
    os.makedirs(os.path.dirname(args.model_save_path) or '.', exist_ok=True)
    np.save(args.model_save_path, best_weights)

    config_path = args.model_save_path.replace('.npy', '.json')
    if 'best_model' in config_path:
        config_path = config_path.replace('best_model', 'best_config')
        
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()
