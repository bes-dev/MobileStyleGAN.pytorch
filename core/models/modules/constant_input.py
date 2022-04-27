import torch
import torch.nn as nn


class ConstantInput(nn.Module):
    def __init__(self, channel, size=(4, 4)):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size[0], size[1]))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out
