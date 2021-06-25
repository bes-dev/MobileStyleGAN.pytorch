import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available(): from .fused_act_cuda import *


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, trace_model=False):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale
        self.trace_model = trace_model

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale, self.trace_model)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5, trace_model=False):
    if input.device.type == "cpu":
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        # TODO: fixed ONNX conversion
        if trace_model:
            return F.leaky_relu(input + bias.view(1, input.size(1)), negative_slope=0.2) * scale
        else:
            return (
                F.leaky_relu(
                    input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
                ) * scale
            )

    elif torch.cuda.is_available():
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
    else:
        raise NotImplemented
