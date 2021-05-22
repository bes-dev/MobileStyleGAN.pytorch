import argparse
import os
import torch
from core.utils import select_weights
from core.models.mapping_network import MappingNetwork
from core.models.synthesis_network import SynthesisNetwork, SynthesisBlock


def extract_mnet(ckpt, ckpt_path):
    ckpt_mnet = select_weights(ckpt["g"], "style.")
    style_dim = ckpt_mnet["1.bias"].size()[0]
    n_layers = len([i for i, _ in enumerate(ckpt_mnet) if f"{i}.bias" in ckpt_mnet])
    mnet = MappingNetwork(style_dim, n_layers)
    mnet.layers.load_state_dict(ckpt_mnet)
    torch.save({
        "params": {"style_dim": style_dim, "n_layers": n_layers},
        "ckpt": mnet.state_dict()
    }, ckpt_path)
    return style_dim


def extract_snet(ckpt, style_dim, ckpt_path):
    convs = select_weights(ckpt["g"], "convs.")
    to_rgbs = select_weights(ckpt["g"], "to_rgbs.")

    blocks = []
    channels = []

    i = 0
    while True:
        conv1 = select_weights(convs, f"{i * 2}.")
        if not len(conv1):
            break
        conv2 = select_weights(convs, f"{i * 2 + 1}.")
        to_rgb = select_weights(to_rgbs, f"{i}.")

        c_in = conv1["conv.weight"].size()[2]
        c_out = conv2["conv.weight"].size()[1]
        channels.append(c_in)
        block = SynthesisBlock(
            c_in,
            c_out,
            style_dim
        )
        block.conv1.load_state_dict(conv1)
        block.conv2.load_state_dict(conv2)
        block.to_rgb.load_state_dict(to_rgb)
        blocks.append(block)
        size = 2 ** (3 + i)
        i += 1
    channels.append(c_out)

    snet = SynthesisNetwork(size, style_dim, channels=channels)
    snet.input.load_state_dict(select_weights(ckpt["g"], "input."))
    snet.conv1.load_state_dict(select_weights(ckpt["g"], "conv1."))
    snet.to_rgb1.load_state_dict(select_weights(ckpt["g"], "to_rgb1."))
    for i, _ in enumerate(snet.layers):
        snet.layers[i].load_state_dict(blocks[i].state_dict())

    torch.save({
        "params": {
            "size": size,
            "style_dim": style_dim,
            "channels": channels
        },
        "ckpt": snet.state_dict()
    }, ckpt_path)


def main(args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    print("extract mapping network")
    style_dim = extract_mnet(ckpt, args.ckpt_mnet)
    print("extract synthesis network")
    extract_snet(ckpt, style_dim, args.ckpt_snet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--ckpt", type=str, help="path to input ckpt")
    parser.add_argument("--ckpt-mnet", type=str, help="path to output mapping_network ckpt")
    parser.add_argument("--ckpt-snet", type=str, help="path to output synthesis_network ckpt")
    args = parser.parse_args()
    main(args)
