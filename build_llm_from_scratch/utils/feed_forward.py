import torch
import torch.nn as nn

# GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0) / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
# Feed forward neural network with GELU activation function
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], cfg['emb_dim'] * 4), # Expands the dimension by a factor of 4
            GELU(),
            nn.Linear(cfg['emb_dim'] * 4, cfg['emb_dim']) # Projects back to the original dimension
        )
    def forward(self, x):
        return self.layers(x)
