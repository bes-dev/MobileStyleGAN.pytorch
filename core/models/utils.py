import torch


class NoiseManager:
    def __init__(self, noise, device, trace_model=False):
        self.device = device
        self.noise_lut = {}
        if noise is not None:
            for i in range(len(noise)):
                if not None in noise:
                    self.noise_lut[noise[i].size(-1)] = noise[i]
        self.trace_model = trace_model

    def __call__(self, size, b=1):
        if self.trace_model:
            return None if b == 1 else [None] * b
        if size in self.noise_lut:
            return self.noise_lut[size]
        else:
            return torch.randn(b, 1, size, size).to(self.device)
