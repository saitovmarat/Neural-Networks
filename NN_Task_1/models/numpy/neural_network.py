import numpy as np
from utils.metrics import relu, relu_derivative, \
                          sigmoid, sigmoid_derivative, \
                          softmax, cross_entropy, \
                          accuracy, mse

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
        
        if self.task == 'classification':
            dZ = cache[f'A{L}'] - y
        else:
            dZ = (cache[f'A{L}'] - y) / batch_size
            
        grads[f'dW{L}'] = dZ.T @ cache[f'A{L-1}'] 
        grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True)

        for layer in reversed(range(1, L)):
            dA = dZ @ self.parameters[f'W{layer+1}']
            if self.activation == 'relu':
                dZ = dA * relu_derivative(cache[f'Z{layer}'])
            elif self.activation == 'sigmoid':
                dZ = dA * sigmoid_derivative(cache[f'Z{layer}']) 
            
            grads[f'dW{layer}'] = dZ.T @ cache[f'A{layer-1}'] 
            grads[f'db{layer}'] = np.sum(dZ, axis=0, keepdims=True)

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
            if self.task == 'classification':
                loss = cross_entropy(y, y_pred)
                acc = accuracy(y, y_pred)
                losses.append(loss)
            else:
                loss = mse(y, y_pred)
                losses.append(loss)
                
            if (epoch+1) % 100 == 0:
                if self.task == 'classification':
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
                    
        return losses
    
    def train_loader(self, data_loader, epochs=100, learning_rate=0.01):
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in data_loader:
                X_np = X_batch.numpy()
                y_np = y_batch.numpy()

                y_pred, cache = self.forward(X_np)
                self.backward(X_np, y_np, cache, learning_rate)

                if self.task == 'classification':
                    loss = cross_entropy(y_np, y_pred)
                else:
                    loss = mse(y_np, y_pred)
                total_loss += loss * X_batch.shape[0]

            avg_loss = total_loss / len(data_loader.dataset)
            losses.append(avg_loss)

            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred


