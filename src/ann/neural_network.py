"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import wandb
from .neural_layer import layer 
from .objective_functions import cross_entropy_loss, cross_entropy_grad, mean_squared_grad, mean_squared_loss
from .optimizers import SGD, Momentum, NAG, RMSprop

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
        if len(hidden_layer_size) == 1 and cli_args.num_layers > 1:
            hidden_layer_size = hidden_layer_size * cli_args.num_layers

        self.learning_rate = cli_args.learning_rate
        self.loss = cli_args.loss
        self.model_save_path = cli_args.model_save_path

        all_dims = [784] + hidden_layer_size + [10]

        for i in range(len(all_dims) - 1):
            in_dim = all_dims[i]
            out_dim = all_dims[i+1]
            
            if i < len(all_dims) - 2:
                name = f"hidden{i+1}"
                activation = cli_args.activation
            else:
                name = "output_layer"
                activation = 'linear' 

            self.layers[name] = layer(
                no_of_neurons = out_dim, 
                no_of_inputs = in_dim,
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
        
        self.weight_decay = cli_args.weight_decay

    
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
        
        grad_W_list = []
        grad_b_list = []

        curr = {key : self.layers[key] for key in reversed(self.layers)}
        for layer_name, l in curr.items():
            error = l.backward_pass(error)
            l.grad_W += (self.weight_decay * l.weights)
            grad_W_list.append(l.grad_W)
            grad_b_list.append(l.grad_b)
        
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        # print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        # print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update()
    
    def train(self, X_train, y_train, epochs = 1, batch_size = 32):
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
            if self.loss == 'mean_squared_error':
                curr_train_loss = mean_squared_loss(y_train, curr_train_pred)
            elif self.loss == 'cross_entropy':
                curr_train_loss = cross_entropy_loss(y_train, curr_train_pred)

            train_acc = self.evaluate(X_train, y_train)
            
            log_dict = {
                "epoch": curr_epoch,
                "train_loss": curr_train_loss,
                "train_accuracy": train_acc
            }
            
            print_str = f"Epoch {curr_epoch} | Train Loss: {curr_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"

            if hasattr(self, 'X_val') and hasattr(self, 'y_val'):
                curr_val_pred = self.forward(self.X_val)
                if self.loss == 'mean_squared_error':
                    curr_val_loss = mean_squared_loss(self.y_val, curr_val_pred)
                elif self.loss == 'cross_entropy':
                    curr_val_loss = cross_entropy_loss(self.y_val, curr_val_pred)
                    
                val_acc = self.evaluate(self.X_val, self.y_val)
                
                log_dict["val_loss"] = curr_val_loss
                log_dict["val_accuracy"] = val_acc
                
                print_str += f" | Val Loss: {curr_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
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
    
    def get_weights(self):
        d = {}
        for i, (name, layer) in enumerate(self.layers.items()):
            d[f"W{i}"] = layer.weights.copy()
            d[f"b{i}"] = layer.biases.copy()
        return d

    def set_weights(self, weight_dict):
        for i, (name, layer) in enumerate(self.layers.items()):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.weights = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.biases = weight_dict[b_key].copy()
