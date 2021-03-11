import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.discriminator import Discriminator
from core.loss.diffaug import get_default_transforms


class GANLoss(nn.Module):
    def __init__(
            self,
            image_size,
            channels_in=3,
            perceptual=False):
        super().__init__()
        self.m = Discriminator(
            size=image_size,
            channels_in=channels_in
        )
        self.perceptual = perceptual
        self.transforms = get_default_transforms()
        self.register_buffer("device_info", torch.ones(1))

    def forward(self, x):
        if self.transforms is not None:
            x = self.transforms(x)
        return self.m(x)

    def loss_g(self, pred, gt):
        labels = torch.ones(pred.size(0), 1).to(self.device_info.device)
        pred_out = self(pred)
        loss_g = self.loss(pred_out["out"], labels)
        loss_p = None
        if self.perceptual:
            with torch.no_grad():
                gt_out = self(gt)
            loss_p = self.loss_perceptual(pred_out, gt_out)
        return loss_g, loss_p

    def loss_d(self, pred, gt):
        # loss real
        labels = torch.ones(gt.size(0), 1).to(self.device_info.device)
        loss_real = self.loss(self(gt)["out"], labels)
        # loss fake
        labels = torch.zeros(pred.size(0), 1).to(self.device_info.device)
        loss_fake = self.loss(self(pred)["out"], labels)
        # loss total
        loss_d = (loss_real + loss_fake) / 2
        return loss_d

    def loss(self, pred, gt):
        return F.binary_cross_entropy(pred, gt)

    def loss_perceptual(self, pred, gt):
        loss = 0
        for p, g in zip(pred["features"], gt["features"]):
            loss += F.l1_loss(p, g)
        return loss
