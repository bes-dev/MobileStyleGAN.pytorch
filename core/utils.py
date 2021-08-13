import json
import torch
import gdown
from addict import Dict


def apply_trace_model_mode(mode=False):
    def _apply_trace_model_mode(m):
        if hasattr(m, 'trace_model'):
            m.trace_model = mode
    return _apply_trace_model_mode


def tensor_to_img(t, normalize=True, range=(-1, 1), to_numpy=True, rgb2bgr=True):
    if normalize:
        t.clamp_(min=range[0], max=range[1])
        t.add_(-range[0]).div_(range[1] - range[0] + 1e-5)
    img = t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    if to_numpy:
        img = img.to('cpu', torch.uint8).numpy()
    if rgb2bgr:
        img = img[:, :, ::-1]
    return img


def download_ckpt(url, name, md5):
    print(f"load pretrained model: {name}...")
    ckpt_path = f"/tmp/{name}"
    gdown.cached_download(url, ckpt_path, md5=md5)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt


def load_cfg(path):
    with open(path) as stream:
        cfg = Dict(json.load(stream))
    return cfg


def save_cfg(path, cfg):
    with open(path, 'w') as stream:
        json.dump(cfg, stream, indent=4)
    return cfg


def select_weights(ckpt, prefix="student."):
    _ckpt = {}
    for k, v in ckpt.items():
        if k.startswith(prefix):
            _ckpt[k.replace(prefix, "")] = v
    return _ckpt


def load_weights(target, source_state):
    from collections import OrderedDict
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        elif k in source_state and v.size() != source_state[k].size():
            print(f"src: {source_state[k].size()}, tgt: {v.size()}")
            new_dict[k] = v
        else:
            print(f"key {k} not loaded...")
            new_dict[k] = v
    target.load_state_dict(new_dict)
