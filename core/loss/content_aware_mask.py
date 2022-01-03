import torch
import torch.nn as nn
from core.models.face_parser import FaceParser


class ContentAwareMask(nn.Module):
    def __init__(self, enable_mask=False):
        super().__init__()
        self.enable_mask = enable_mask
        self.m = FaceParser()

    def forward(self, x):
        if self.enable_mask:
            return self.m(x)
        else:
            return torch.ones(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
