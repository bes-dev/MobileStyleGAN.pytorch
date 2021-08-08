import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out


class EqualDWConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # create conv
        self.weight_dw = nn.Parameter(
            torch.randn(in_channel, 1, kernel_size, kernel_size)
        )
        self.weight_permute = nn.Parameter(
            torch.randn(out_channel, in_channel, 1, 1)
        )
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

    def forward(self, input):
        weight = self.weight_dw.transpose(0, 1) * self.weight_permute
        out = F.conv2d(
            input,
            weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out
