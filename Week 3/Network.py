import numpy as np

class Network:
    def __init__(self, layers, learning_rate=0.001, dense_activation="sigmoid", 
                 output_activation="sigmoid", loss="mse"):
        """
        list layers: Enter a list with elements denoting the number of neurons in each layer. \
        The first layer must correspond to the number of inputs and final layer the number of outputs
        
        str dense_activation & output_activation: Output activation is used for final layer and dense activation \
        for other layers. Currently supports sigmoid, tanh, linear and step
        
        str loss: Currently supporting mse and log_loss
        
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.learning_rate = learning_rate
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.loss_func = loss
        self.weights = [np.random.normal(size=[layers[i], layers[i+1]]) for i in range(len(layers) - 1)]
        self.biases = [np.random.normal(size=layers[i + 1]) for i in range(len(layers) - 1)]
    
    # external functions
    
    def predict(self, X):
        return np.array([self._forward_pass(x) for x in X])
    
    def loss(self, X, y):
        if self.loss_func == "mse":
            return np.mean((self.predict(X) - y) ** 2)
        
        if self.loss_func == "log_loss":
            return -np.mean(np.sum(y * np.log(self.predict(X))))
        
    def accuracy(self, X, y):
        return np.mean(np.argmax(self.predict(X), axis=1) == np.argmax(y, axis=1))
    
    def train(self, X, y, X_val, y_val, epochs=300, verbose=False):
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []
        for i in range(epochs):
            self._train_one_epoch(X, y)
            train_loss.append(self.loss(X, y))
            train_accuracy.append(self.accuracy(X, y))
            val_loss.append(self.loss(X_val, y_val))
            val_accuracy.append(self.accuracy(X_val, y_val))
            if verbose and (i + 1) % 10 == 0:
                print("Epoch: ", i + 1)
                print("Train loss: ", np.round(train_loss[-1], 4))
                print("Train accuracy: ", np.round(train_accuracy[-1], 4))
                print("Validation loss: ", np.round(val_loss[-1], 4))
                print("Validation accuracy: ", np.round(val_accuracy[-1], 4))
                print("\n")
        
        return np.array(train_loss), np.array(train_accuracy), np.array(val_loss), np.array(val_accuracy)
        
    # internal functions

    def _forward_pass(self, x):
        # forward pass calculates a single prediction
        for i in range(self.num_layers - 2):
            z = x.dot(self.weights[i]) + self.biases[i]
            x = self._act_func(z, self.dense_activation)
            
        z = x.dot(self.weights[-1]) + self.biases[-1]
        x = self._act_func(z, self.output_activation)
        return x
    
    def _backward_pass(self, x, y):
        # backward pass calculates the derivative for a single observation
        activations = [x]
        zs = []
        for i in range(self.num_layers - 2):
            zs.append(activations[-1].dot(self.weights[i]) + self.biases[i])
            activations.append(self._act_func(zs[-1], self.dense_activation))
        
        zs.append(activations[-1].dot(self.weights[-1]) + self.biases[-1])
        activations.append(self._act_func(zs[-1], self.output_activation))
        
        nabla_w = [np.zeros_like(self.weights[i]) for i in range(self.num_layers - 1)]
        nabla_b = [np.zeros_like(self.biases[i]) for i in range(self.num_layers - 1)]
        
        delta = self._loss_d(activations[-1], y) * self._act_func_d(zs[-1], self.output_activation)
        nabla_w[-1] = activations[-2].reshape(-1, 1).dot(delta.reshape(-1, 1).T)
        nabla_b[-1] = delta
        
        for i in range(self.num_layers - 2):
            delta = self.weights[-(i + 1)].dot(delta) * self._act_func_d(zs[-(i + 2)], self.dense_activation)
            nabla_w[-(i + 2)] = activations[-(i + 3)].reshape(-1, 1).dot(delta.reshape(-1, 1).T)
            nabla_b[-(i + 2)] = delta
        
        return nabla_w, nabla_b
    
    def _train_one_epoch(self, X, y):
        n = len(X)
        nabla_w = [np.zeros_like(self.weights[i]) for i in range(self.num_layers - 1)]
        nabla_b = [np.zeros_like(self.biases[i]) for i in range(self.num_layers - 1)]
        for x_vec, y_vec in zip(X, y):
            step_w, step_b = self._backward_pass(x_vec, y_vec)
            for i in range(self.num_layers - 1):
                nabla_w[i] += step_w[i] / n
                nabla_b[i] += step_b[i] / n
        
        for i in range(self.num_layers - 1):
            self.weights[i] -= nabla_w[i] * self.learning_rate
            self.biases[i] -= nabla_b[i] * self.learning_rate
        
    def _act_func(self, x_vec, func):
        if func == "sigmoid":
            return 1 / (1 + np.exp(-x_vec))
        
        if func == "linear":
            return x_vec
        
        if func == "step":
            return np.where(x_vec > 0, 1, 0)
        
        if func == "tanh":
            return np.tanh(x_vec)
        
    def _act_func_d(self, x_vec, func):
        if func == "sigmoid":
            return self._act_func(x_vec, func) * (1 - self._act_func(x_vec, func))
        
        if func == "linear" or func == "step":
            return np.ones_like(x_vec)
        
        if func == "tanh":
            return 1 - np.tanh(x_vec) ** 2
    
    def _loss_d(self, a, y):
        if self.loss_func == "mse":
            return a - y
        
        if self.loss_func == "log_loss":
            return -y / a