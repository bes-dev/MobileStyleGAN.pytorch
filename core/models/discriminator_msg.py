import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .modules import *

class ConvLayer(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=1, padding=0, stride=1, bias=True, activate=True):
        super().__init__()
        self.blur = Blur() if stride == 2 else nn.Sequential()
        self.conv = EqualConv2d(channels_in, channels_out, kernel_size, padding=padding, stride=stride, bias=bias)
        self.act = nn.LeakyReLU(0.2) if activate else None

    def forward(self, x):
        x = self.blur(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x) * (2 ** 0.5)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels_in=3, **kwargs):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(inp, oup, 3, stride, 1, bias=True)),
                nn.LeakyReLU(0.2)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                spectral_norm(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True)),
                nn.LeakyReLU(0.2),
                # pw
                spectral_norm(nn.Conv2d(inp, oup, 1, 1, 0, bias=True)),
                nn.LeakyReLU(0.2),
            )

        self.model = nn.Sequential(
            conv_bn(channels_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 2),
        )

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(1024 + 1, 1024, 3, padding=1)
        self.final_linear = nn.Sequential(
            EqualLinear(1024 * 4 * 4, 1024, activation=True),
            EqualLinear(1024, 1),
        )

    def forward(self, x):
        for m in self.model:
            x = m(x)
            print(x.size())
        out = self.minibatch_discrimination(x, self.stddev_group, self.stddev_feat)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        out = self.final_linear(out)
        return out


    @staticmethod
    def minibatch_discrimination(x, stddev_group, stddev_feat):
        out = x
        batch, channel, height, width = out.shape
        group = min(batch, stddev_group)
        stddev = out.view(group, -1, stddev_feat, channel // stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        return out


if __name__ == "__main__":
    disc = Discriminator(3)
    print(disc)

    # import time
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(disc, (3, 1024, 1024), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # var = torch.randn(1, 3, 1024, 1024)
    # out = disc(var)
