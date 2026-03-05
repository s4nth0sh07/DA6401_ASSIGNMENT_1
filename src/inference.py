"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load_data, process_data
from ann.neural_network import NeuralNetwork
from ann.objective_functions import cross_entropy_loss, mean_squared_loss

def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    # Required Inference Arguments
    parser.add_argument('--model_path', type=str, default='models/best_model.npy', help='Path to saved model weights')
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('-sz', '--hidden_size', type=int, default=[128], help='Number of neurons in each hidden layer', nargs='+')
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'])
    
    # Dummy arguments required to safely initialize the NeuralNetwork class without crashing
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-o', '--optimizer', type=str, default='sgd')
    parser.add_argument('-w_i', '--weight_init', type=str, default='random')
    parser.add_argument('--model_save_path', type=str, default='')
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    # .item() extracts the dictionary from the 0-d numpy array it was saved as
    return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
    """
    # 1. Forward pass to get logits/probabilities
    logits = model.forward(X_test)
    
    # 2. Calculate the specific loss
    if model.loss == 'mean_squared_error':
        loss = mean_squared_loss(y_test, logits)
    else:
        loss = cross_entropy_loss(y_test, logits)
        
    # 3. Convert one-hot encoded vectors back to standard class indices (0 through 9)
    y_true_idx = np.argmax(y_test, axis=1)
    y_pred_idx = np.argmax(logits, axis=1)
    
    # 4. Calculate metrics using scikit-learn (Macro average treats all classes equally)
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    precision = precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
    recall = recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
    f1 = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
    
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
    
    print(f"Loading {args.dataset} dataset...")
    # We must load the training data so process_data can properly normalize the test data!
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_data(args.dataset)
    _, X_test, _, y_test, _, _ = process_data(X_train_raw, X_test_raw, y_train_raw, y_test_raw)
    
    print("Rebuilding architecture...")
    model = NeuralNetwork(cli_args=args)
    
    print(f"Loading weights from {args.model_path}...")
    weights_dict = load_model(args.model_path)
    
    # Inject the saved weights into the newly created layers
    for name, layer in model.layers.items():
        layer.weights = weights_dict[f"{name}_W"]
        layer.biases = weights_dict[f"{name}_b"]
        
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
    
    return results


if __name__ == '__main__':
    main()