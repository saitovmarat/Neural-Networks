import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        image = image.reshape(28, 28)
        image = image / 255.0  
        image = image.reshape(-1)  
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)