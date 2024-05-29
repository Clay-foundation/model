import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class ChesapeakeDataset(Dataset):
    def __init__(self, chip_dir, label_dir, metadata, platform):
        self.chip_dir = Path(chip_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )

        self.chips = [chip_path.name for chip_path in self.chip_dir.glob("*.npy")]
        self.labels = [re.sub("_naip-new_", "_lc_", chip) for chip in self.chips]

    def create_transforms(self, mean, std):
        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        chip_name = self.chip_dir / self.chips[idx]
        label_name = self.label_dir / self.labels[idx]

        chip = np.load(chip_name).astype(np.float32)
        label = np.load(label_name)

        # Remap labels
        # 1 = water
        # 2 = tree canopy / forest
        # 3 = low vegetation / field
        # 4 = barren land
        # 5 = impervious (other)
        # 6 = impervious (road)
        # 15 = no data
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 15: 6}
        remapped_label = np.vectorize(label_mapping.get)(label)

        if self.transform:
            chip = self.transform(chip)

        sample = {
            "pixels": self.transform(torch.from_numpy(chip)),
            "label": torch.from_numpy(label[0]),
            "time": torch.zeros(4),
            "latlon": torch.zeros(4),
        }
        return sample


class ChesapeakeDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_chip_dir,
        train_label_dir,
        val_chip_dir,
        val_label_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
    ):
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.train_label_dir = train_label_dir
        self.val_chip_dir = val_chip_dir
        self.val_label_dir = val_label_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform

    def setup(self, stage=None):
        if stage in {"fit", None}:
            self.trn_ds = ChesapeakeDataset(
                self.train_chip_dir, self.train_label_dir, self.metadata, self.platform
            )
            self.val_ds = ChesapeakeDataset(
                self.val_chip_dir, self.val_label_dir, self.metadata, self.platform
            )

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
