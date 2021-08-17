import torch
import torch.nn as nn


class WaveletHaar2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5

    def forward(self, x, mode="forward"):
        assert mode in ["forward", "inverse"]
        return getattr(self, f"_transform_{mode}")(x)

    def _transform_forward(self, x):
        a0 = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        a1 = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        a2 = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        a3 = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        return torch.cat([a0,a1,a2,a3], axis=1)

    def _transform_inverse(self, x):
        size = [x.shape[0], x.shape[1] // 4, x.shape[2] * 2, x.shape[3] * 2]
        out = torch.zeros(size, dtype=x.dtype, device=x.device)
        f = lambda i: x[:, size[1] * i : size[1] * (i + 1)]
        out[:,:,0::2,0::2] = self.alpha * (f(0) + f(1) + f(2) + f(3))
        out[:,:,0::2,1::2] = self.alpha * (f(0) + f(1) - f(2) - f(3))
        out[:,:,1::2,0::2] = self.alpha * (f(0) - f(1) + f(2) - f(3))
        out[:,:,1::2,1::2] = self.alpha * (f(0) - f(1) - f(2) + f(3))
        return out
