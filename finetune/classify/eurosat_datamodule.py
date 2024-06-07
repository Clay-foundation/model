import lightning as L
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT as TGEuroSAT
from torchvision.transforms import v2


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

    def __init__(self, root, split, bands, transforms, download):
        super().__init__(root, split, bands, transforms, download)

    def __getitem__(self, index):
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
        sample = {
            "pixels": image,
            "label": label,
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for lat/lon information
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class EuroSATDataModule(L.LightningDataModule):
    """
    Data module for loading and transforming the EuroSAT dataset.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of workers for data loading.
        metadata_path (str): Path to the metadata file for normalization
        statistics.
    """

    def __init__(self, batch_size, num_workers, metadata_path):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        metadata = Box(yaml.safe_load(open(metadata_path)))["sentinel-2-l2a"]
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

    def setup(self, stage=None):
        """
        Setup the datasets for training and validation.

        Args:
            stage (str): Stage of the training process ('fit', 'validate',
            etc.).
        """
        if stage in {"fit", None}:
            self.trn_ds = EuroSAT(
                root="data",
                split="train",
                bands=[
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B08A",
                    "B11",
                    "B12",
                ],
                transforms=self.trn_tfm,
                download=True,
            )
            self.val_ds = EuroSAT(
                root="data",
                split="val",
                bands=[
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B08A",
                    "B11",
                    "B12",
                ],
                transforms=self.val_tfm,
                download=True,
            )

    def train_dataloader(self):
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

    def val_dataloader(self):
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
