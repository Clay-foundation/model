import random
from pathlib import Path
from typing import List

import lightning as L
import numpy as np
import rasterio as rio
import torch
from torch.utils.data import DataLoader, Dataset


class ClayDataset(Dataset):
    def __init__(self, chips_path: List[Path], transform=None):
        super().__init__()
        self.chips_path = chips_path
        self.transform = transform

    def read_chip(self, chip_path):
        ts = chip_path.parent.name
        year, month, day = map(np.float16, ts.split("-"))

        chip = rio.open(chip_path)
        bounds = chip.bounds
        centroid_x = (bounds.left + bounds.right) / 2
        centroid_y = (bounds.top + bounds.bottom) / 2

        return {
            "pixels": chip.read(),
            "timestep": (year, month, day),
            "latlon": (centroid_x, centroid_y),
        }

    def __getitem__(self, idx):
        chip_path = self.chips_path[idx]
        cube = self.read_chip(chip_path)

        if self.transform:
            cube["pixels"] = torch.as_tensor(data=cube["pixels"])
            cube["latlon"] = torch.as_tensor(data=cube["latlon"])
            cube["timestep"] = torch.as_tensor(data=cube["timestep"])

        return cube

    def __len__(self):
        return len(self.chips_path)


class ClayDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path = Path("data/"),
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        chips_path = list(self.data_dir.glob("**/*.tif"))
        random.shuffle(chips_path)
        split_ratio = 0.8
        split = int(len(chips_path) * split_ratio)

        self.trn_ds = ClayDataset(chips_path[:split], transform=True)
        self.val_ds = ClayDataset(chips_path[split:], transform=True)

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
