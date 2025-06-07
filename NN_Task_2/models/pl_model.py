import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

class NeuralNetworkPL(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', task='regression'):
        super().__init__()
        self.task = task
        self.activation_name = activation
        self.save_hyperparameters()

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError("Unsupported activation function")

        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(hidden_sizes): 
                layers.append(self._get_activation())

        self.net = nn.Sequential(*layers)

    def _get_activation(self):
        if self.activation_name == 'relu':
            return nn.ReLU()
        elif self.activation_name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.task == 'classification':
            loss = F.cross_entropy(y_hat, y.long().squeeze())
        else:
            loss = F.mse_loss(y_hat, y)
            
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.task == 'classification':
            loss = F.cross_entropy(y_hat, y.long().squeeze())
        else:
            loss = F.mse_loss(y_hat, y)
            
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.task == 'classification':
            loss = F.cross_entropy(y_hat, y.long().squeeze())
        else:
            loss = F.mse_loss(y_hat, y)
            
        self.log("test_loss", loss)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self(X_tensor)
            probs = F.softmax(logits, dim=1).numpy()
        return probs