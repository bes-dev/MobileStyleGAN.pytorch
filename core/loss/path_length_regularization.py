import torch
import torch.nn as nn


class PathLengthRegularization(nn.Module):
    def __init__(self, decay=0.01):
        super().__init__()
        self.decay = decay
        self.register_buffer("path_mean", torch.tensor(0))

    def forward(self, fake, latent):
        h, w = fake.size()[2:]
        noise = torch.randn_like(fake).div_( (h * w) ** 0.5)
        grad, = torch.autograd.grad(
            outputs=(fake * noise).sum(),
            inputs-latent,
            create_graph=True
        )
        path_length = torch.sqrt(grad.pow(2).sum(2).mean(1))
        self.path_mean = self.path_mean + decay * (path_length.mean() - self.path_mean)
        path_loss = (path_length - self.path_mean).pow(2).mean()
        return {
            "path_loss": path_loss,
            "path_length": path_length
        }
