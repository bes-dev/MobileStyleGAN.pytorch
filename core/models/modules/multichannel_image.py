import torch
import torch.nn as nn
from .modulated_conv2d import ModulatedConv2d


class MultichannelIamge(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=1
    ):
        super().__init__()
        self.conv = ModulatedConv2d(channels_in, channels_out, style_dim, kernel_size, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))

    def forward(self, hidden, style):
        out = self.conv(hidden, style)
        out = out + self.bias
        return out
