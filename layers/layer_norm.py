import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, n_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)
