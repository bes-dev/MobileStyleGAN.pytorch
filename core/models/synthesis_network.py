import torch
import torch.nn as nn
from .modules.legacy import *


class SynthesisBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            style_dim,
            kernel_size=3,
            blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()
        self.conv1 = StyledConv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=True,
            blur_kernel=blur_kernel
        )
        self.conv2 = StyledConv(
            out_channel,
            out_channel,
            kernel_size,
            style_dim,
            blur_kernel=blur_kernel
        )
        self.to_rgb = ToRGB(out_channel, style_dim)

    def forward(self, hidden, style, noise=[None, None]):
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])
        rgb = self.to_rgb(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, rgb


class SynthesisNetwork(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            blur_kernel=[1, 3, 3, 1],
            channels = [512, 512, 512, 512, 512, 256, 128, 64, 32]
    ):
        super().__init__()
        self.size = size
        self.style_dim = style_dim

        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv(
            channels[0], channels[0], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(channels[0], style_dim, upsample=False)

        self.layers = nn.ModuleList()
        in_channel = channels[0]
        for out_channel in channels[1:]:
            self.layers.append(
                SynthesisBlock(
                    in_channel,
                    out_channel,
                    style_dim,
                    3,
                    blur_kernel=blur_kernel
                )
            )
            in_channel = out_channel

        self.upsample = Upsample(blur_kernel)

    def forward(self, style, noise=None):
        out = {"noise": [], "rgb": [], "img": None}

        hidden = self.input(style)
        if noise is None:
            _noise = torch.randn(1, 1, hidden.size(-1), hidden.size(-1)).to(style.device)
        else:
            _noise = noise[0]
        out["noise"].append(_noise)
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=_noise)
        img = self.to_rgb1(hidden, style if style.ndim == 2 else style[:, 1, :])
        out["rgb"].append(img)

        for i, m in enumerate(self.layers):
            shape = [2, 1, 1, 2 ** (i + 3), 2 ** (i + 3)]
            if noise is None:
                _noise = torch.randn(*shape).to(style.device)
            else:
                _noise = noise[i + 1]
            out["noise"].append(_noise)
            _style = style if style.ndim == 2 else style[:, 3*i+1:3*i+4, :]
            hidden, rgb = m(hidden, style, _noise)
            out["rgb"].append(rgb)
            img = self.upsample(img) + rgb

        out["img"] = img
        return out

    def wsize(self):
        return len(self.layers) * 3 + 2
