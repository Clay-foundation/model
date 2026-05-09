from typing import Any, cast

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT as TGEuroSAT
from torchvision.transforms import v2

from claymodel.metadata import load_metadata_yaml

S2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]


class EuroSAT(TGEuroSAT):
    """
    Subclass of TGEuroSAT to customize the dataset loading and
    transformations.

    Args:
        root (str): Root directory of the dataset.
        split (str): Dataset split to use ('train' or 'val').
        bands (list): List of spectral bands to use.
        transforms (callable): Transformations to apply to the samples.
        download (bool): If true, downloads the dataset.
    """

    def __init__(
        self,
        root: str,
        split: str,
        bands: list[str],
        transforms: Any,
        download: bool,
    ) -> None:
        super().__init__(root, split, bands, transforms, download)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Override the __getitem__ method to apply custom transformations.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image tensor, label, and
            additional metadata.
        """
        image, label = self._load_image(index)

        image = torch.index_select(image, dim=0, index=self.band_indices).float()
        if self.transforms is not None:
            sample = {
                "pixels": image,
                "label": label,
                "time": torch.zeros(4),
                "latlon": torch.zeros(4),
            }
            return cast(
                "dict[str, torch.Tensor]",
                self.transforms(sample),  # ty: ignore[missing-argument]
            )

        return {
            "pixels": image,
            "label": label,
            "time": torch.zeros(4),
            "latlon": torch.zeros(4),
        }


class EuroSATDataModule(L.LightningDataModule):
    """
    Data module for loading and transforming the EuroSAT dataset.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of workers for data loading.
        metadata_path (str): Path to the metadata file for normalization
        statistics.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        metadata_path: str,
        data_dir: str = "data",
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

        metadata = load_metadata_yaml(metadata_path)["sentinel-2-l2a"]
        mean = list(metadata.bands.mean.values())
        std = list(metadata.bands.std.values())

        self.trn_tfm = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.Normalize(mean, std),
            ]
        )
        self.val_tfm = v2.Compose([v2.Normalize(mean, std)])

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the datasets for training and validation.

        Args:
            stage (str): Stage of the training process ('fit', 'validate',
            etc.).
        """
        if stage in {"fit", None}:
            self.trn_ds = EuroSAT(
                root=self.data_dir,
                split="train",
                bands=S2_BANDS,
                transforms=self.trn_tfm,
                download=True,
            )
            self.val_ds = EuroSAT(
                root=self.data_dir,
                split="val",
                bands=S2_BANDS,
                transforms=self.val_tfm,
                download=True,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
