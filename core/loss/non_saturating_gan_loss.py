import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from core.models.discriminator import Discriminator
from .diffaug import get_default_transforms


class R1Regularization(nn.Module):
    def __init__(self, r1_gamma=10):
        super().__init__()
        # r1 params
        self.r1_gamma = r1_gamma

    def forward(self, x, x_pred):
        grad_x = grad(outputs=x_pred.sum(), inputs=x, create_graph=True)[0]
        grad_penalty = (grad_x.view(grad_x.size(0), -1).norm(2, dim=1) ** 2).mean()
        r1_loss = 0.5 * self.r1_gamma * grad_penalty
        return r1_loss


class NonSaturatingGANLoss(nn.Module):
    def __init__(
            self,
            image_size,
            channels_in=3,
            r1_gamma=10):
        super().__init__()
        # discriminator
        self.m = Discriminator(
            size=image_size,
            channels_in=channels_in,
            activate=False
        )
        # diffaugs
        self.transforms = get_default_transforms()
        # r1 regularization
        self.r1_reg = R1Regularization(r1_gamma)

    def forward(self, x, diffaug_mode=True):
        if self.transforms is not None and diffaug_mode:
            x = self.transforms(x)
        return self.m(x)

    def loss_g(self, fake, *args, **kwargs):
        fake_loss = F.softplus( -self(fake, True)["out"] ).mean()
        return fake_loss

    def loss_d(self, fake, real):
        fake, real = fake.detach(), real.detach()
        # fake loss
        fake_pred = self(fake, True)["out"]
        fake_loss = F.softplus( fake_pred ).mean()
        # real loss
        real_pred = self(real, True)["out"]
        real_loss = F.softplus( -real_pred ).mean()
        # total loss
        total_loss = fake_loss + real_loss
        return total_loss

    def reg_d(self, real):
        # r1 regularization for real imgs
        real.requires_grad = True
        real_pred = self(real, False)["out"]
        r1_loss = self.r1_reg(real, real_pred)
        return r1_loss
