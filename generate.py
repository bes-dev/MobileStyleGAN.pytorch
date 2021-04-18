import argparse
import os
import cv2
import torch
import numpy as np
from core.utils import load_cfg, load_weights, tensor_to_img
from core.distiller import Distiller
from core.model_zoo import model_zoo
from tqdm import tqdm

@torch.no_grad
def main(args):
    cfg = load_cfg(args.cfg)
    distiller = Distiller(cfg)
    if args.ckpt is not None:
        ckpt = model_zoo(args.ckpt)
        load_weights(distiller, ckpt["state_dict"])

    distiller = distiller.to(args.device)
    for i in tqdm(range(args.n_batches)):
        var = torch.randn(args.batch_size, distiller.mapping_net.style_dim).to(args.device)
        img_s = distiller(var, truncated=args.truncated)
        for j in range(img_s.size(0)):
            cv2.imwrite(
                os.path.join(args.output_path, f"{i*args.batch_size + j}.png"),
                tensor_to_img(img_s[j].cpu())
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--device", type=str, default="cpu", help="select device for inference")
    parser.add_argument("--cfg", type=str, default="configs/mobile_stylegan_ffhq.json", help="path to config file")
    parser.add_argument("--ckpt", type=str, default="mobilestylegan_ffhq.ckpt", help="path to checkpoint")
    parser.add_argument("--truncated", action='store_true', help="use truncation mode")
    parser.add_argument("--output-path", type=str, default="./", help="path to store images")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size")
    parser.add_argument("--n-batches", type=int, default=5000, help="number of batches")
    args = parser.parse_args()
    main(args)
