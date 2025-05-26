import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):    
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_pred):
    batch_size = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / batch_size

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', task='regression'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.task = task
        self.parameters = {}
        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in range(1, len(self.layer_sizes)):
            scale = np.sqrt(2.0 / (self.layer_sizes[layer-1] + self.layer_sizes[layer]))
            self.parameters[f'W{layer}'] = np.random.randn(self.layer_sizes[layer], self.layer_sizes[layer-1]) * scale
            self.parameters[f'b{layer}'] = np.zeros((1, self.layer_sizes[layer]))

    def forward(self, X):
        cache = {'A0': X}
        L = len(self.parameters) // 2

        for layer in range(1, L):
            W = self.parameters[f'W{layer}']
            b = self.parameters[f'b{layer}']
            Z = cache[f'A{layer-1}'] @ W.T + b
            if self.activation == 'relu':
                A = relu(Z)
            elif self.activation == 'sigmoid':
                A = sigmoid(Z)
                
            cache[f'Z{layer}'] = Z
            cache[f'A{layer}'] = A

        WL = self.parameters[f'W{L}']
        bL = self.parameters[f'b{L}']
        ZL = cache[f'A{L-1}'] @ WL.T + bL
        if self.task == 'classification':
            AL = softmax(ZL)
        else:
            AL = ZL 
        cache[f'Z{L}'] = ZL
        cache[f'A{L}'] = AL

        return AL, cache

    def backward(self, X, y, cache, learning_rate=0.01):
        grads = {}
        batch_size = X.shape[0]
        L = len(self.parameters) // 2
        
        dZ = cache[f'A{L}'] - y 
        grads[f'dW{L}'] = (cache[f'A{L-1}'].T @ dZ) / batch_size
        grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / batch_size

        for layer in reversed(range(1, L)):
            W_next = self.parameters[f'W{layer+1}']
            dA = dZ @ W_next
            if self.activation == 'relu':
                dZ = dA * relu_derivative(cache[f'Z{layer}'])
            elif self.activation == 'sigmoid':
                dZ = dA * sigmoid_derivative(cache[f'Z{layer}']) 
            
            grads[f'dW{layer}'] = (cache[f'A{layer-1}'].T @ dZ) / batch_size
            grads[f'db{layer}'] = np.sum(dZ, axis=0, keepdims=True) / batch_size

        for layer in range(1, L+1):
            self.parameters[f'W{layer}'] -= learning_rate * grads[f'dW{layer}']
            self.parameters[f'b{layer}'] -= learning_rate * grads[f'db{layer}']

    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred, cache = self.forward(X_batch)
                self.backward(X_batch, y_batch, cache, learning_rate)

            y_pred = self.predict(X)
            loss = cross_entropy(y, y_pred) if self.task == 'classification' else mse(y, y_pred)
            losses.append(loss)
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        return losses

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred