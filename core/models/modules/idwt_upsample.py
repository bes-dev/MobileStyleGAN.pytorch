import torch
import torch.nn as nn
from .modulated_conv2d import ModulatedConv2d
from .idwt import DWTInverse

class IDWTUpsaplme(nn.Module):
    def __init__(
            self,
            channels_in,
            style_dim,
    ):
        super().__init__()
        self.channels = channels_in // 4
        assert self.channels * 4 == channels_in
        # upsample
        self.idwt = DWTInverse(mode='zero', wave='db1')
        # modulation
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)

    def forward(self, x, style):
        b, _, h, w = x.size()
        x = self.modulation(style).view(b, -1, 1, 1) * x
        low = x[:, :self.channels]
        high = x[:, self.channels:]
        high = high.view(b, self.channels, 3, h, w)
        x = self.idwt((low, [high]))
        return x
