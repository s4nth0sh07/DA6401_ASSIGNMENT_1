import numpy as np

def accuracy_score(y_true_idx, y_pred_idx):
    return np.mean(y_true_idx == y_pred_idx)

def precision_score(y_true_idx, y_pred_idx, num_classes=10):
    precisions = []
    for c in range(num_classes):
        tp = np.sum((y_pred_idx == c) & (y_true_idx == c))
        fp = np.sum((y_pred_idx == c) & (y_true_idx != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)
    return np.mean(precisions)

def recall_score(y_true_idx, y_pred_idx, num_classes=10):
    recalls = []
    for c in range(num_classes):
        tp = np.sum((y_pred_idx == c) & (y_true_idx == c))
        fn = np.sum((y_pred_idx != c) & (y_true_idx == c))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    return np.mean(recalls)

def f1_score(y_true_idx, y_pred_idx, num_classes=10):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred_idx == c) & (y_true_idx == c))
        fp = np.sum((y_pred_idx == c) & (y_true_idx != c))
        fn = np.sum((y_pred_idx != c) & (y_true_idx == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return np.mean(f1s)