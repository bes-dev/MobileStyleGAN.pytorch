import torch
import torch.nn as nn
from .modules.legacy import PixelNorm, EqualLinear


class MappingNetwork(nn.Module):
    def __init__(
            self,
            style_dim,
            n_layers,
            lr_mlp=0.01
    ):
        super().__init__()
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_layers):
            layers.append(
                EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
