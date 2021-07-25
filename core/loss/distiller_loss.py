import torch
import torch.nn as nn
import torch.nn.functional as F
from core.loss.non_saturating_gan_loss import NonSaturatingGANLoss
from core.loss.perceptual_loss import PerceptualLoss
from pytorch_wavelets import DWTInverse, DWTForward
import math


class DistillerLoss(nn.Module):
    def __init__(
            self,
            discriminator_size,
            perceptual_size=256,
            loss_weights={"l1": 1.0, "l2": 1.0, "loss_p": 1.0, "loss_g": 0.5}
    ):
        super().__init__()
        # l1/l2 loss
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        # perceptual_loss
        self.perceptual_loss = PerceptualLoss(perceptual_size)
        # gan loss
        self.gan_loss = NonSaturatingGANLoss(image_size=int(discriminator_size))
        # loss weights
        self.loss_weights = loss_weights
        # utils
        self.dwt = DWTForward(J=1, mode='zero', wave='db1')
        self.idwt = DWTInverse(mode="zero", wave="db1")

    def loss_g(self, pred, gt):
        # l1/l2 loss
        loss = {"l1": 0, "l2": 0}
        pred_rgbs, gt_rgbs = [], []
        for _pred in pred["freq"]:
            _pred_rgb = self.dwt_to_img(_pred)
            _gt_rgb = F.interpolate(gt["img"], size=_pred_rgb.size(-1), mode='bilinear', align_corners=True)
            _gt_freq = self.img_to_dwt(_gt_rgb)
            loss["l1"] += self.l1_loss(_pred_rgb, _gt_rgb)
            loss["l2"] += self.l2_loss(_pred_rgb, _gt_rgb)
            loss["l1"] += self.l1_loss(_pred, _gt_freq)
            loss["l2"] += self.l2_loss(_pred, _gt_freq)
            pred_rgbs.append(_pred_rgb)
            gt_rgbs.append(_gt_rgb)
        pred_rgbs = pred_rgbs[::-1]
        gt_rgbs = gt_rgbs[::-1]
        # perceptual_loss
        loss["loss_p"] = self.perceptual_loss(pred["img"], gt["img"])
        # gan loss
        # loss["loss_g"] = self.gan_loss.loss_g(pred["img"], gt["img"])
        loss["loss_g"] = self.gan_loss.loss_g(pred_rgbs, gt_rgbs)

        # total loss
        loss["loss"] = 0
        for k, w in self.loss_weights.items():
            if loss[k] is not None:
                loss["loss"] += w * loss[k]
            else:
                del loss[k]
        return loss

    def loss_d(self, pred, gt):
        loss = {}
        # loss["loss"] = loss["loss_d"] = self.gan_loss.loss_d(pred["img"].detach(), gt["img"])
        loss["loss"] = loss["loss_d"] = self.gan_loss.loss_d(
            self.make_freq_pyramid(pred["freq"]),
            self.make_img_pyramid(gt["img"])
            # pred["img"].detach(),
            # gt["img"]
        )
        return loss

    def reg_d(self, real):
        out = {}
        # out["loss"] = out["d_reg"] = self.gan_loss.reg_d(real["img"])
        out["loss"] = out["d_reg"] = self.gan_loss.reg_d(
            self.make_img_pyramid(real["img"])
        )
        return out

    def make_img_pyramid(self, img):
        rgbs = [img]
        log_size = int(math.log(img.size(-1), 2))
        for i in range(log_size, 3, -1):
            size = 2 ** (i - 1)
            rgb = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
            rgbs.append(rgb.detach())
        return rgbs

    def make_freq_pyramid(self, freqs):
        rgbs = []
        for f in freqs:
            rgbs.append(self.dwt_to_img(f).detach())
        return rgbs[::-1]

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))
