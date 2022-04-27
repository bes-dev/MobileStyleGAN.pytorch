import torch
import torch.nn as nn
from .utils import NoiseManager
from .modules import StyledConv2d, \
    ConstantInput, \
    MultichannelIamge, \
    ModulatedDWConv2d, \
    MobileSynthesisBlock, \
    DWTInverse


class MobileSynthesisNetwork(nn.Module):
    def __init__(
            self,
            style_dim,
            channels = [512, 512, 512, 512, 512, 256, 128, 64],
            input_shape = (4, 4)
    ):
        super().__init__()
        self.style_dim = style_dim

        self.input = ConstantInput(channels[0], input_shape)
        self.conv1 = StyledConv2d(
            channels[0],
            channels[0],
            style_dim,
            kernel_size=3
        )
        self.to_img1 = MultichannelIamge(
            channels_in=channels[0],
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module=ModulatedDWConv2d
                )
            )
            channels_in = channels_out

        self.idwt = DWTInverse(mode="zero", wave="db1")
        self.register_buffer("device_info", torch.zeros(1))
        self.trace_model = False

    def forward(self, style, noise=None):
        out = {"noise": [], "freq": [], "img": None}
        noise = NoiseManager(noise, self.device_info.device, self.trace_model)

        hidden = self.input(style)
        out["noise"].append(noise(hidden.size(-1)))
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=out["noise"][-1])
        img = self.to_img1(hidden, style if style.ndim == 2 else style[:, 1, :])
        out["freq"].append(img)

        for i, m in enumerate(self.layers):
            out["noise"].append(noise(2 ** (i + 3), 2))
            _style = style if style.ndim == 2 else style[:, m.wsize()*i + 1: m.wsize()*i + m.wsize() + 1, :]
            hidden, freq = m(hidden, _style, noise=out["noise"][-1])
            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])
        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2
