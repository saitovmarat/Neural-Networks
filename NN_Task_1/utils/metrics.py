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
    return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-15)  

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_pred):
    batch_size = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / batch_size

def accuracy(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(y_true_labels == y_pred_labels)