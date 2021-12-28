import os
# random
import random
# pytorch
import torch
import torch.nn as nn
import pytorch_lightning as pl
# dataset
from core.dataset import NoiseDataset
# teacher model
from core.model_zoo import model_zoo
from core.models.mapping_network import MappingNetwork
from core.models.synthesis_network import SynthesisNetwork
# student model
from core.models.mobile_synthesis_network import MobileSynthesisNetwork
# loss
from core.loss.distiller_loss import DistillerLoss
# evaluation network
from core.models.inception_v3 import load_inception_v3
# evaluation metric
from piq import KID
# utils
from core.utils import apply_trace_model_mode


class Distiller(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg.trainer

        # teacher model
        print("load mapping network...")
        mapping_net_ckpt = model_zoo(**cfg.teacher.mapping_network)
        self.mapping_net = MappingNetwork(**mapping_net_ckpt["params"]).eval()
        self.mapping_net.load_state_dict(mapping_net_ckpt["ckpt"])
        print("load synthesis network...")
        synthesis_net_ckpt = model_zoo(**cfg.teacher.synthesis_network)
        self.synthesis_net = SynthesisNetwork(**synthesis_net_ckpt["params"]).eval()
        self.synthesis_net.load_state_dict(synthesis_net_ckpt["ckpt"])
        # student network
        self.student = MobileSynthesisNetwork(
            style_dim=self.mapping_net.style_dim,
            channels=synthesis_net_ckpt["params"]["channels"][:-1]
        )

        # dataset
        self.wsize = self.student.wsize()
        self.trainset = NoiseDataset(batch_size=self.cfg.batch_size, **cfg.trainset)
        self.valset = NoiseDataset(batch_size=self.cfg.batch_size, **cfg.valset)

        #compute style_mean
        self.register_buffer(
            "style_mean",
            self.compute_mean_style(self.mapping_net.style_dim, wsize=self.wsize, batch_size=4096)
        )

        # loss
        self.loss = DistillerLoss(
            discriminator_size=self.synthesis_net.size,
            **cfg.distillation_loss
        )

        # evaluator
        self.kid = KID()
        self.inception = load_inception_v3()

        # device info
        self.register_buffer("device_info", torch.tensor(1))

    def _log_loss(self, loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, exclude=["loss"]):
        for k, v in loss.items():
            if not k in exclude:
                self.log(k, v, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        mode = self.opt_to_mode[optimizer_idx]
        if mode == "g":
            loss = self.generator_step(batch, batch_nb)
            self._log_loss(loss)
        elif mode == "d":
            loss = self.discriminator_step(batch, batch_nb)
            self._log_loss(loss)
        return {"loss": loss["loss"]}

    def validation_step(self, batch, batch_nb):
        # compute inception_v3 features
        style, gt = self.make_sample(batch)
        pred = self.student(style, noise=gt["noise"])
        pred_inc = self.inception(pred["img"])[0].view(style.size(0), -1)
        gt_inc = self.inception(gt["img"])[0].view(style.size(0), -1)
        # compute val_loss
        pred = self.student(style, noise=gt["noise"])
        loss = self.loss.loss_g(pred, gt)
        return {"pred": pred_inc, "gt": gt_inc, "loss_val": loss["loss"]}

    def validation_epoch_end(self, outputs):
        # TODO: add all_gather for distributed mode
        # agregate kid_val
        pred, gt = [], []
        for x in outputs:
            pred.append(x["pred"])
            gt.append(x["gt"])
        pred = torch.cat(pred, axis=0)
        gt = torch.cat(gt, axis=0)
        kid = self.kid.compute_metric(pred, gt)
        self.log("kid_val", kid, prog_bar=True)
        # agregate val_loss
        loss = torch.stack([x['loss_val'] for x in outputs]).mean()
        self.log("loss_val", loss, prog_bar=True)

    def generator_step(self, batch, batch_nb):
        style, gt = self.make_sample(batch)
        pred = self.student(style, noise=gt["noise"])
        loss = self.loss.loss_g(pred, gt)
        return loss

    def discriminator_step(self, batch, batch_nb):
        style, gt = self.make_sample(batch)
        with torch.no_grad():
            pred = self.student(style, noise=gt["noise"])
        if self.global_step % self.cfg.reg_d_interval != 0:
            loss = self.loss.loss_d(pred, gt)
        else:
            loss = self.loss.reg_d(gt)
            loss["loss"] *= self.cfg.reg_d_interval
        return loss

    @torch.no_grad()
    def make_sample(self, batch):
        def make_style():
            var = torch.randn(self.cfg.batch_size, self.mapping_net.style_dim).to(self.device_info.device)
            style = self.mapping_net(var)
            return style

        coin = random.random()
        if coin >= self.cfg.stylemix_p[1]:
            style = self.mapping_net(batch["noise"]).unsqueeze(1).repeat(1, self.wsize, 1)
        elif coin >= self.cfg.stylemix_p[0] and coin < self.cfg.stylemix_p[1]:
            style_a, style_b = make_style(), make_style()
            inject_index = random.randint(1, self.wsize - 1)
            style_a = style_a.unsqueeze(1).repeat(1, inject_index, 1)
            style_b = style_b.unsqueeze(1).repeat(1, self.wsize - inject_index, 1)
            style = torch.cat([style_a, style_b], dim=1)
        else:
            var = torch.randn(self.wsize, self.mapping_net.style_dim).to(self.device_info.device)
            style = self.mapping_net(var).view(1, self.wsize, self.mapping_net.style_dim)

        if self.cfg.truncated:
            style = self.style_mean + 0.5 * (style - self.style_mean)

        gt = self.synthesis_net(style)
        return style, gt

    @torch.no_grad()
    def compute_mean_style(self, style_dim, wsize=1, batch_size=4096):
        style = self.mapping_net(torch.randn(4096, self.mapping_net.style_dim)).mean(0, keepdim=True)
        if wsize != 1:
            style = style.unsqueeze(1).repeat(1, wsize, 1)
        return style

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.trainset.batch_size, num_workers=self.cfg.num_workers, shuffle=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, batch_size=self.valset.batch_size, num_workers=self.cfg.num_workers, shuffle=False)

    def configure_optimizers(self):
        opts = []
        self.opt_to_mode = {}
        mode = self.cfg.mode.split(',')
        for i, mode in enumerate(self.cfg.mode.split(',')):
            if mode == "g":
                print("setup generator train mode on...")
                opts.append(torch.optim.Adam(self.student.parameters(), lr=self.cfg.lr_student))
                self.opt_to_mode[i] = "g"
            elif mode == "d":
                print("setup discriminator train mode on...")
                opts.append(torch.optim.Adam(self.loss.gan_loss.parameters(), lr=self.cfg.lr_gan))
                self.opt_to_mode[i] = "d"
        return opts, []

    def forward(self, var, truncated=False, generator="student"):
        var = var.to(self.device_info.device)
        style = self.mapping_net(var)
        if truncated:
            style = self.style_mean + 0.5 * (style - self.style_mean)
        if generator == "student":
            img = self.student(style)["img"]
        else:
            img = self.synthesis_net(style)["img"]
        return img

    def simultaneous_forward(self, var, truncated=False):
        var = var.to(self.device_info.device)
        style = self.mapping_net(var)
        if truncated:
            style = self.style_mean + 0.5 * (style - self.style_mean)
        out_t = self.synthesis_net(style)
        img_s = self.student(style, noise=out_t["noise"])["img"]
        return img_s, out_t["img"]

    def to_onnx(self, output_dir, w_plus=False):
        class Wrapper(nn.Module):
            def __init__(
                    self,
                    synthesis_network,
                    style_tmp
            ):
                super().__init__()
                self.m = synthesis_network
                self.noise = self.m(style_tmp)["noise"]

            def forward(self, style):
                return self.m(style, noise=self.noise)["img"]

        print("prepare style...")
        if not w_plus:
            var = torch.randn(1, self.mapping_net.style_dim).to(self.device_info.device)
            style = self.mapping_net(var)
        else:
            var = torch.randn(self.wsize, self.mapping_net.style_dim).to(self.device_info.device)
            style = self.mapping_net(var)
            style = style.view(1, self.wsize, -1)

        print("convert mapping network...")
        self.mapping_net.apply(apply_trace_model_mode(True))
        torch.onnx.export(
            self.mapping_net,
            (var,),
            os.path.join(output_dir, "MappingNetwork.onnx"),
            input_names = ['var'],
            output_names = ['style'],
            verbose=True
        )

        print("convert synthesis network...")
        self.student.apply(apply_trace_model_mode(True))
        torch.onnx.export(
            Wrapper(self.student, style),
            (style,),
            os.path.join(output_dir, "SynthesisNetwork.onnx"),
            input_names = ['style'],
            output_names = ['img'],
            verbose=True
        )

    def to_coreml(self, output_dir, w_plus=False):
        import coremltools as ct

        print("prepare style...")
        if not w_plus:
            var = torch.randn(1, self.mapping_net.style_dim).to(self.device_info.device)
            style = self.mapping_net(var)
        else:
            var = torch.randn(self.wsize, self.mapping_net.style_dim).to(self.device_info.device)
            style = self.mapping_net(var)
            style = style.view(1, self.wsize, -1)

        print("convert mapping network...")
        self.mapping_net.apply(apply_trace_model_mode(True))
        mapping_net_trace = torch.jit.trace(self.mapping_net, var)
        mapping_net_coreml = ct.convert(
            mapping_net_trace,
            inputs=[ct.TensorType(name="var", shape=var.shape)]
        )
        mapping_net_coreml.save(os.path.join(output_dir, "MappingNetwork.mlmodel"))

        print("convert synthesis network...")
        self.student.apply(apply_trace_model_mode(True))
        # initialize noise buffers
        self.student(style)
        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, style):
                return self.m(style)["img"]

        synthesis_net = Wrapper(self.student)
        synthesis_net_trace = torch.jit.trace(synthesis_net, style)
        synthesis_net_coreml = ct.convert(
            synthesis_net_trace,
            inputs=[ct.TensorType(name="style", shape=style.shape)]
        )
        synthesis_net_coreml.save(os.path.join(output_dir, "SynthesisNetwork.mlmodel"))
