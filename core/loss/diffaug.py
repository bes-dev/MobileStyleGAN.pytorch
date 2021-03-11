import torch
import torch.nn as nn
import kornia

def get_default_transforms():
    return nn.Sequential(
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomAffine(
            translate=(0.1, 0.3),
            scale=(0.7, 1.2),
            degrees=(-20, 20)
        ),
        kornia.augmentation.RandomErasing()
    )
