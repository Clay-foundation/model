import yaml
from pathlib import Path

import torch
import lightning as L
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
from einops import rearrange
from PIL import Image
from box import Box
from torchvision.transforms import v2
from torchgeo.datasets import EuroSAT
from torch.utils.data import DataLoader, Dataset

from torchgeo.datasets import EuroSAT as TGEuroSAT

class EuroSAT(TGEuroSAT):
    def __init__(self, root, split, bands, transforms, download):
        super().__init__(root, split, bands, transforms, download)

    def __getitem__(self, index):
        image, label = self._load_image(index)

        image = torch.index_select(image, dim=0, index=self.band_indices).float()
        sample = {'pixels': image, 'label': label, 'time': torch.zeros(4), 'latlon': torch.zeros(4)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

class EuroSATDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, metadata_path):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        metadata = Box(yaml.safe_load(open(metadata_path)))["sentinel-2-l2a"]
        mean = list(metadata.bands.mean.values())
        std = list(metadata.bands.std.values())
        
        self.trn_tfm = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.Normalize(mean, std)
        ])
        self.val_tfm = v2.Compose([
            v2.Normalize(mean, std)
        ])

    def setup(self, stage=None):
        if stage in {"fit", None}:
            self.trn_ds = EuroSAT(root="data", split="train", bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B11", "B12"], transforms=self.trn_tfm, download=True)
            self.val_ds = EuroSAT(root="data", split="val", bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B11", "B12"], transforms=self.val_tfm, download=True)

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size = self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )