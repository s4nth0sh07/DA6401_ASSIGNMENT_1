"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from utils.data_loader import load_data, process_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ann.neural_network import NeuralNetwork
from ann.objective_functions import cross_entropy_loss, mean_squared_loss

def parse_arguments():
    """
    Parse command-line arguments for inference.
    Identical to train.py to satisfy updated assignment constraints.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-l', '--loss', type=str, default='mean_squared_error', choices=['mean_squared_error', 'cross_entropy'])
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-nhl', '--num_layers', type=int, default=1)
    parser.add_argument('-sz', '--hidden_size', type=str, default=["128"], nargs='+')
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier'])
    parser.add_argument('-w_p', '--wandb_project', type=str, default='ASSIGNMENT-1')
    parser.add_argument('--model_save_path', type=str, default='src/best_model.npy')
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
    """
    logits = model.forward(X_test)
    
    if model.loss == 'mean_squared_error':
        loss = mean_squared_loss(y_test, logits)
    else:
        loss = cross_entropy_loss(y_test, logits)
        
    y_true_idx = np.argmax(y_test, axis=1)
    y_pred_idx = np.argmax(logits, axis=1)
    
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    precision = precision_score(y_true_idx, y_pred_idx)
    recall = recall_score(y_true_idx, y_pred_idx)
    f1 = f1_score(y_true_idx, y_pred_idx)
    
    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """
    Main inference function.
    """
    args = parse_arguments()
    if args.hidden_size:
        raw_sz = " ".join(args.hidden_size) if isinstance(args.hidden_size, list) else args.hidden_size
        args.hidden_size = [int(x.strip('[], ')) for x in raw_sz.replace(',', ' ').split()]
    
    print(f"Loading {args.dataset} dataset...")
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_data(args.dataset)
    _, X_test, _, y_test, _, _ = process_data(X_train_raw, X_test_raw, y_train_raw, y_test_raw)
    
    print("Rebuilding architecture...")
    model = NeuralNetwork(cli_args=args)
    
    print(f"Loading weights from {args.model_save_path}...")
    weights_dict = load_model(args.model_save_path)
    
    model.set_weights(weights_dict)
    
    print("Running evaluation on test set...\n")
    results = evaluate_model(model, X_test, y_test)
    
    print("="*30)
    print("      EVALUATION RESULTS")
    print("="*30)
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print("="*30)
    
    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()