import math
import random
from pathlib import Path
from typing import List

import lightning as L
import numpy as np
import rasterio as rio
import torch
from pyproj import Transformer as ProjTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class ClayDataset(Dataset):
    def __init__(self, chips_path: List[Path], transform=None):
        super().__init__()
        self.chips_path = chips_path
        self.transform = transform

    def normalize_timestamp(self, ts):
        year, month, day = map(np.float16, ts.split("-"))
        year_radians = 2 * math.pi * (year - 2012) / (2030 - 2012)  # years 2012-2030
        month_radians = 2 * math.pi * (month - 1) / 11
        day_radians = (
            2 * math.pi * (day - 1) / 30
        )  # Assuming a 31-day month for simplicity

        # Normalize using sine and cosine
        year = math.atan2(math.cos(year_radians), math.sin(year_radians))
        month = math.atan2(math.cos(month_radians), math.sin(month_radians))
        day = math.atan2(math.cos(day_radians), math.sin(day_radians))

        return year, month, day

    def normalize_latlon(self, lat, lon):
        lat_radians = math.radians(lat)
        lon_radians = math.radians(lon)

        # Apply sine and cosine
        lat = math.sin(lat_radians)
        lon = math.atan2(
            math.cos(lon_radians),
            math.sin(lon_radians),
        )
        return lat, lon

    def read_chip(self, chip_path):
        # read timestep & normalize
        ts = chip_path.parent.name
        year, month, day = self.normalize_timestamp(ts)

        chip = rio.open(chip_path)

        # read lat,lon from UTM to WGS84 & normalize
        bounds = chip.bounds
        crs = chip.crs
        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.top + bounds.bottom) / 2
        tfmer = ProjTransformer.from_crs(crs, "epsg:4326", always_xy=True)
        lat, lon = tfmer.transform(cx, cy)
        lat, lon = self.normalize_latlon(lat, lon)

        return {
            "pixels": chip.read(),
            "timestep": (year, month, day),
            "latlon": (lat, lon),
        }

    def __getitem__(self, idx):
        chip_path = self.chips_path[idx]
        cube = self.read_chip(chip_path)

        # remove nans and convert to tensor
        cube["pixels"] = torch.nan_to_num(torch.as_tensor(data=cube["pixels"]), nan=0.0)
        cube["latlon"] = torch.as_tensor(data=cube["latlon"])
        cube["timestep"] = torch.as_tensor(data=cube["timestep"])

        if self.transform:
            # convert to float16 and normalize
            cube["pixels"] = self.transform(cube["pixels"])

        return cube

    def __len__(self):
        return len(self.chips_path)


class ClayDataModule(L.LightningDataModule):
    MEAN = [
        518.393981,
        670.384338,
        583.347534,
        961.506958,
        1903.755737,
        2138.707519,
        2238.332031,
        2273.117919,
        1413.791137,
        808.279968,
        0.033653,
        0.135196,
        536.390136,
    ]

    STD = [
        876.523559,
        918.090148,
        981.493835,
        1001.560729,
        1256.656372,
        1346.299072,
        1414.495483,
        1392.251342,
        918.297912,
        605.479919,
        0.048188,
        0.380075,
        630.602233,
    ]

    def __init__(
        self,
        data_dir: Path = Path("data/"),
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tfm = v2.Compose(
            [
                v2.Normalize(mean=self.MEAN, std=self.STD),
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        chips_path = list(self.data_dir.glob("**/*.tif"))
        random.shuffle(chips_path)
        split_ratio = 0.8
        split = int(len(chips_path) * split_ratio)

        self.trn_ds = ClayDataset(chips_path[:split], transform=self.tfm)
        self.val_ds = ClayDataset(chips_path[split:], transform=self.tfm)

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
