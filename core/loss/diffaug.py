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


class DiffAug:
    def __init__(self):
        self.transforms = get_default_transforms()

    def apply(self, img):
        return self.transforms(img)

    def apply_to_pyramid(self, pyramid):
        for t in self.transforms:
            params = t.generate_parameters(
                self.get_normalized_batch_size(pyramid[0].size())
            )
            for i in range(len(pyramid)):
                pyramid[i] = t.apply_transform(pyramid[i], params)
        return pyramid

    def get_normalized_batch_size(self, size):
        _size = list(size)
        _size[2] = _size[3] = 1
        return _size
