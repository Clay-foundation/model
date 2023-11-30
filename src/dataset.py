from pathlib import Path

import lightning as L
import rasterio as rio
import torch
from torch.utils.data import DataLoader, Dataset


class ClayDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.chips_path = list(data_dir.glob("**/*.tif"))
        self.transform = transform

    def read_chip(self, chip_path):
        ts = chip_path.parent.name
        year, month, day = map(int, ts.split("-"))

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
        data_dir: Path,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.trn_ds = ClayDataset(self.data_dir, transform=True)

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
