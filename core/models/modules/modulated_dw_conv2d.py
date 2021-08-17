import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedDWConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True,
            dense_mode=True,
    ):
        super().__init__()
        # create conv
        self.weight_dw = nn.Parameter(
            torch.randn(channels_in, 1, kernel_size, kernel_size)
        )
        self.weight_permute = nn.Parameter(
            torch.randn(channels_out, channels_in, 1, 1)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        self.modulation_act = nn.Tanh()
        # create demodulation parameters
        self.demodulate = demodulate
        # some service staff
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.dense_mode = dense_mode

    def forward(self, x, style):
        if self.dense_mode:
            return self._forward_dense(x, style)
        else:
            return self._forward_dw(x, style)

    def _forward_dense(self, x, style):
        b, c, h, w = x.size()
        modulation = self.get_modulation(style).view(b, 1, -1, 1, 1)
        weight = modulation * (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        if self.demodulate:
            demodulation = self.get_demodulation(style).view(1, -1, 1, 1, 1)
            weight = weight * demodulation
        weight = weight.view(b * self.channels_out, self.channels_in, self.kernel_size, self.kernel_size)
        x = x.view(1, b * c, h, w)
        x = F.conv2d(x, weight, padding=self.padding, groups=b)
        x = x.view(b, -1, x.size(2), x.size(3))
        return x

    def _forward_dw(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        if self.demodulate:
            style = self.modulation_act(style)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        b, _ = style.size()
        w = (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        norm = torch.rsqrt((self.scale * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)
