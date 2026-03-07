import wandb
import numpy as np
from keras.datasets import mnist

def log_data_exploration():
    wandb.init(project="ASSIGNMENT-1", name="data_exploration_table")

    (X_train, y_train), _ = mnist.load_data()

    columns = ["Class Label", "Image 1", "Image 2", "Image 3", "Image 4", "Image 5"]
    exploration_table = wandb.Table(columns=columns)

    for class_label in range(10):
        indices = np.where(y_train == class_label)[0]
        
        sample_indices = indices[:5]
        
        images = [wandb.Image(X_train[idx]) for idx in sample_indices]
        
        exploration_table.add_data(str(class_label), *images)

    wandb.log({"MNIST_Class_Distribution": exploration_table})
    wandb.finish()
    print("Table successfully logged to W&B!")

if __name__ == '__main__':
    log_data_exploration()