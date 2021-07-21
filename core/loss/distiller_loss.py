import torch
import torch.nn as nn
import torch.nn.functional as F
from core.loss.non_saturating_gan_loss import NonSaturatingGANLoss
from core.loss.perceptual_loss import PerceptualLoss
from pytorch_wavelets import DWTInverse, DWTForward


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
        for _pred in pred["freq"]:
            _pred_rgb = self.dwt_to_img(_pred)
            _gt_rgb = F.interpolate(gt["img"], size=_pred_rgb.size(-1), mode='bilinear', align_corners=True)
            _gt_freq = self.img_to_dwt(_gt_rgb)
            loss["l1"] += self.l1_loss(_pred_rgb, _gt_rgb)
            loss["l2"] += self.l2_loss(_pred_rgb, _gt_rgb)
            loss["l1"] += self.l1_loss(_pred, _gt_freq)
            loss["l2"] += self.l2_loss(_pred, _gt_freq)
        # perceptual_loss
        loss["loss_p"] = self.perceptual_loss(pred["img"], gt["img"])
        # gan loss
        loss["loss_g"] = self.gan_loss.loss_g(pred["img"], gt["img"])

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
        loss["loss"] = loss["loss_d"] = self.gan_loss.loss_d(pred["img"].detach(), gt["img"])
        return loss

    def reg_d(self, real):
        out = {}
        out["loss"] = out["d_reg"] = self.gan_loss.reg_d(real["img"])
        return out

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
