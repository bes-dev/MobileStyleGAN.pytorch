import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bisenet import BiSeNet
from core.utils import download_ckpt


class Normalize(nn.Module):
    """ Implementation of differentiable normalization of the input tensor
    Arguments:
        mean (list<float>): per channel mean.
        std (list<float>): per channel mean.
        input_range (tuple<float, float>): range of the input values.
    """
    def __init__(
            self,
            mean: typing.List[float],
            std: typing.List[float],
            input_range: typing.Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        # input range
        self.input_range = input_range
        # project input_range -> [0, 1]
        range_shift = input_range[0]
        range_scale = input_range[1] - input_range[0] + 1e-5
        # prepare mean
        mean = torch.Tensor(mean).view(1, -1, 1, 1)
        mean = range_shift + range_scale * mean
        self.register_buffer("mean", mean)
        # prepare std
        std = torch.Tensor(std).view(1, -1, 1, 1)
        std = range_scale * std
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor.
        Arguments:
            x (torch.Tensor): input tensor.
        Returns:
            y (torch.Tensor): normalized tensor.
        """
        # clamp range
        x = torch.clamp(x, self.input_range[0], self.input_range[1])
        # normalize
        x = (x - self.mean) / (self.std + 1e-5)
        return x


class FaceParser(nn.Module):
    def __init__(
            self,
            n_classes = 19,
            input_size = (512, 512),
            ckpt = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
    ):
        super().__init__()
        self.input_size = input_size
        self.preprocessor = Normalize(
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225),
            input_range=(-1.0, 1.0)
        )
        self.m = BiSeNet(n_classes=n_classes).eval()
        if ckpt is not None:
            self.m.load_state_dict(download_ckpt(ckpt, "face_parser.ckpt", None))

    def forward(self, x):
        input_size = x.size()[2:]
        x = F.interpolate(x, self.input_size, mode="bilinear", align_corners=False)
        x = self.preprocessor(x)
        mask = (self.m(x)[0].argmax(1) != 0).float()
        mask = F.interpolate(mask.unsqueeze(1), input_size, mode="bilinear", align_corners=False)
        mask = (mask > 0.5)
        return mask
