import glob
import os
import torch
import cv2
import albumentations as aug
from albumentations.pytorch.transforms import ToTensorV2


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, name_regexp="*.png", transforms=None):
        super().__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.names = glob.glob(os.path.join(self.data_path, name_regexp))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.names[idx])), cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return {"img": img}

    @staticmethod
    def get_default_transforms(image_size=1024):
        return aug.Compose([
            aug.Resize(p=1, height=image_size, width=image_size),
            aug.Normalize(),
            ToTensorV2()
        ])


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, emb_size, batch_size, n_batches, precompute=False):
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_batches = n_batches
        if precompute:
            self.embs = torch.randn(batch_size * n_batches, emb_size)

    def __len__(self):
        return self.batch_size * self.n_batches

    def __getitem__(self, idx):
        out = {}
        out["noise"] = torch.randn(self.emb_size) if self.embs is None else self.embs[idx]
        return out
