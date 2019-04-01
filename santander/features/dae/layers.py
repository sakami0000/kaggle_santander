import torch
from torch import nn


class BatchSwapNoise(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size()) < self.prob
            shift = (torch.floor(torch.rand(x.size()) * x.size(0)).type(torch.LongTensor) *
                        (mask.type(torch.LongTensor) * x.size(1))).view(-1)
            idx = torch.add(torch.arange(x.nelement()), shift)
            idx[idx >= x.nelement()] = idx[idx >= x.nelement()] - x.nelement()
            swapped = x.view(-1)[idx].view(x.size())
            return swapped
        else:
            return x
