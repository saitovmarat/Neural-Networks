import pytorch_lightning as pl
import torch.nn as nn
import torch 
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class ResNetClassifier(pl.LightningModule):
    def __init__(self, output_size):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, output_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.long().squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.long().squeeze())
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.long().squeeze())
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs