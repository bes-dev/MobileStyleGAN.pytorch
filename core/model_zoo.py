import json
import torch
from core.utils import download_ckpt

def model_zoo(name, zoo_path="configs/model_zoo.json"):
    zoo = json.load(open(zoo_path))
    if name in zoo:
        ckpt = download_ckpt(**zoo[name])
    else:
        ckpt = torch.load(name, map_location="cpu")
    return ckpt
