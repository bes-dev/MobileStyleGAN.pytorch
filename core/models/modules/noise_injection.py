import torch
import torch.nn as nn


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.trace_model = False

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        if not hasattr(self, "noise") and self.trace_model:
            self.register_buffer("noise", noise)
        if self.trace_model:
            noise = self.noise
        return image + self.weight * noise
