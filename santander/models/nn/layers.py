import torch
from torch import nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class DenseModule(nn.Module):
    def __init__(self, input_size, output_size, activation='relu', dropout_rate=0.1):
        super(DenseModule, self).__init__()
        activations = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'celu': nn.CELU,
            'rrelu': nn.RReLU,
            'leakyrelu': nn.LeakyReLU,
            'prelu': nn.PReLU,
            'swish': Swish
        }
        assert activation in activations.keys()

        if dropout_rate:
            normalization = nn.Dropout(dropout_rate)
        else:
            normalization = nn.BatchNorm1d(output_size)

        self.dense = nn.Sequential(
            nn.Linear(input_size, output_size),
            activations[activation](),
            normalization
        )

    def forward(self, x):
        out = self.dense(x)
        return out
