"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import wandb
from .neural_layer import layer 
from .objective_functions import cross_entropy_loss, cross_entropy_grad, mean_squared_grad, mean_squared_loss
from .optimizers import SGD, Momentum, NAG, RMSprop, Adam, Nadam

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = {}
        hidden_layer_size = cli_args.hidden_size
        self.learning_rate = cli_args.learning_rate
        self.loss = cli_args.loss
        self.model_save_path = cli_args.model_save_path

        self.layers['hidden1'] = layer(
            no_of_neurons = hidden_layer_size[0],
            no_of_inputs = 784,
            weight_init = cli_args.weight_init,
            activation = cli_args.activation
        )

        for i in range(len(hidden_layer_size)):
            input, output = hidden_layer_size[i], 0
            if i == len(hidden_layer_size) - 1:
                activation = 'softmax'
                name = 'output_layer'
                output = 10
            else:
                activation = cli_args.activation
                name = f"hidden{i + 2}"
                output =  hidden_layer_size[i + 1]

            
            self.layers[name] = layer(
                no_of_neurons = output, 
                no_of_inputs = input,
                weight_init = cli_args.weight_init,
                activation = activation
            )
        
        optimization_func = cli_args.optimizer.lower()
        self.optimizer = None
        match optimization_func:
            case 'sgd':
                self.optimizer = SGD(self.layers, self.learning_rate)
            case 'momentum':
                self.optimizer = Momentum(self.layers, self.learning_rate, b = 0.9)
            case 'nag':
                self.optimizer = NAG(self.layers, learning_rate = self.learning_rate, b = 0.9)
            case 'rmsprop':
                self.optimizer = RMSprop(self.layers, learning_rate = self.learning_rate, b = 0.999, e = 1e-8)
            case 'adam':
                self.optimizer = Adam(self.layers, learning_rate = self.learning_rate, b1 = 0.9, b2 = 0.999, e = 1e-8)
            case 'nadam':
                self.optimizer = Nadam(self.layers, learning_rate = self.learning_rate, b1 = 0.9, b2 = 0.999, e = 1e-8)

    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        res = X
        for l in self.layers.values():
            res = l.forward_pass(res)
        return res
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        if self.loss == 'mean_squared_error':
            error = mean_squared_grad(y_true, y_pred)
        elif self.loss == 'cross_entropy':
            error = cross_entropy_grad(y_true, y_pred)
        
        curr = {key : self.layers[key] for key in reversed(self.layers)}
        for layer_name, l in curr.items():
            error = l.backward_pass(error)
        
        grads_W = {name : l.grad_W for name, l in self.layers.items()}
        grads_b = {name : l.grad_b for name, l in self.layers.items()}

        return grads_W, grads_b
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update()
    
    def train(self, X_train, y_train, epochs, batch_size, X_val = None, y_val = None):
        """
        Train the network for specified epochs.
        """
        no_of_images = X_train.shape[0]
        no_of_batches =int(np.ceil(no_of_images / batch_size))
        rem = no_of_images % batch_size
        curr_epoch = 1
        while epochs != 0:
            index = np.random.permutation(no_of_images)
            X_train_new, y_train_new = X_train[index], y_train[index]
            b = 0
            for i in range(no_of_batches):
                if rem and i == no_of_batches - 1:
                    X_curr, y_curr = X_train_new[b : b + rem], y_train_new[b : b + rem]
                else:
                    X_curr, y_curr = X_train_new[b : b + batch_size], y_train_new[b : b + batch_size]
                
                y_pred = self.forward(X_curr)
                self.backward(y_curr, y_pred)
                self.update_weights()
                b += batch_size
            
            curr_train_pred = self.forward(X_train)
            curr_train_loss = None
            curr_val_pred = self.forward(X_val)
            curr_val_loss = None
            if self.loss == 'mean_squared_error':
                curr_train_loss = mean_squared_loss(y_train, curr_train_pred)
                curr_val_loss = mean_squared_loss(y_val, curr_val_pred)
            elif self.loss == 'cross_entropy':
                curr_train_loss = cross_entropy_loss(y_train, curr_train_pred)
                curr_val_loss = cross_entropy_loss(y_val, curr_val_pred)

            accuracy = ((self.evaluate(X_val, y_val), self.evaluate(X_train, y_train)))

            log_dict = {
                "epoch": curr_epoch,
                "train_loss": curr_train_loss,
                "train_accuracy": accuracy[1],
                "val_loss": curr_val_loss,
                "val_accuracy": accuracy[0]
            }
            print_str = f"Epoch {curr_epoch} | Train Loss: {curr_train_loss:.4f} | Train Acc: {accuracy[1]*100:.2f}%"
            print(print_str)
            print_str = f"Epoch {curr_epoch} | Validation Loss: {curr_val_loss:.4f} | Validation Acc: {accuracy[0]*100:.2f}%"
            print(print_str)

            if wandb.run is not None:
                wandb.log(log_dict)

            curr_epoch += 1
            epochs -= 1
        
        model = {}
        for name, layer in self.layers.items():
            model[f"{name}_W"] = layer.weights
            model[f"{name}_b"] = layer.biases

        np.save(self.model_save_path, model)


    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        y_pred = self.forward(X)
        expected_op, actual_op = np.argmax(y, axis = 1), np.argmax(y_pred, axis = 1)
        res = 0
        res = np.sum(expected_op ==  actual_op)
        
        return res / expected_op.shape[0]
