"""
DataModule for the BioMasters dataset for a regression task.

BioMassters: A Benchmark Dataset for Forest Biomass Estimation using
Multi-modal Satellite Time-series https://nascetti-a.github.io/BioMasster/

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a regression model.

Citation:

Andrea Nascetti, Ritu Yadav, Kirill Brodt, Qixun Qu, Hongwei Fan, Yuri
Shendryk, Isha Shah, and Christine Chung, BioMassters: A Benchmark Dataset
for Forest Biomass Estimation using Multi-modal Satellite Time-series,
Thirty-seventh Conference on Neural Information Processing Systems Datasets
and Benchmarks Track, 2023, https://openreview.net/forum?id=hrWsIC4Cmz
"""

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from tifffile import imread
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from claymodel.metadata import load_metadata_yaml


class BioMastersDataset(Dataset):
    """
    Dataset class for the BioMasters regression dataset.

    Assumes band order
    vv, vh, vv, vh, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
    """

    def __init__(
        self,
        chip_dir: str | Path,
        label_dir: str | Path,
        metadata: Any,
    ) -> None:
        self.chip_dir = Path(chip_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata

        # Load statistics from Clay metadata
        s2_mean = list(metadata["sentinel-2-l2a"].bands.mean.values())
        s2_std = list(metadata["sentinel-2-l2a"].bands.std.values())

        # Duplicate the S1 statistics so that the asc/desc orbit data
        # is handled correctly
        self.transform = self.create_transforms(
            mean=s2_mean,
            std=s2_std,
        )
        # Load chip and label file names
        self.chips = [chip_path.name for chip_path in self.chip_dir.glob("*.npz")]
        print(f"Found {len(self.chips)} chips to process for {chip_dir}")

    def create_transforms(self, mean: list[float], std: list[float]) -> v2.Compose:
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
            ],
        )

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # ty: ignore[invalid-method-override]
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        chip_name = self.chip_dir / self.chips[idx]
        label_name = self.label_dir / (chip_name.stem.split("_")[-1] + "_agbm.tif")

        chip = np.load(chip_name)["cube"][2:, ...].astype("float32")
        label = imread(label_name).astype("float32")
        label = np.expand_dims(label, 0)

        return {
            "pixels": self.transform(torch.from_numpy(chip)),
            "label": torch.from_numpy(label),
            "time": torch.zeros(4),
            "latlon": torch.zeros(4),
        }


class BioMastersDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        train_label_dir (str): Directory containing training labels.
        val_chip_dir (str): Directory containing validation image chips.
        val_label_dir (str): Directory containing validation labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(  # noqa: PLR0913
        self,
        train_chip_dir: str | Path,
        train_label_dir: str | Path,
        val_chip_dir: str | Path,
        val_label_dir: str | Path,
        metadata_path: str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.train_label_dir = train_label_dir
        self.val_chip_dir = val_chip_dir
        self.val_label_dir = val_label_dir
        self.metadata = load_metadata_yaml(metadata_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        if stage in {"fit", None}:
            self.trn_ds = BioMastersDataset(
                self.train_chip_dir,
                self.train_label_dir,
                self.metadata,
            )
            self.val_ds = BioMastersDataset(
                self.val_chip_dir,
                self.val_label_dir,
                self.metadata,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
