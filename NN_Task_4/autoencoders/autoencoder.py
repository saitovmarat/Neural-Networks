import pytorch_lightning as pl
import torch.nn as nn
import torch

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.decoder(self.encoder(x))

    def get_latent(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)  
        x_recon = self(x)
        loss = self.loss_fn(x_recon, x)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_recon = self(x)
        loss = self.loss_fn(x_recon, x)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)