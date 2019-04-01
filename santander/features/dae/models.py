import torch
from torch import nn

from .layers import BatchSwapNoise


class DenoisedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise = BatchSwapNoise(0.15)
        self.encoder = nn.Sequential(
            nn.Linear(200, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 50)
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.Linear(100, 200)
        )

    def forward(self, xb): 
        encoder = self.encoder(self.noise(xb))
        decoder = self.decoder(encoder)
        return decoder
