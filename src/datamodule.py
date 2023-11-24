"""
LightningDataModule to load Earth Observation data from GeoTIFF files using
rasterio.
"""
import lightning as L
import numpy as np
import rasterio
import torch
import torchdata


# %%
def _array_to_torch(filepath: str) -> torch.Tensor:
    """
    Read a GeoTIFF file using rasterio into a numpy.ndarray, and convert it
    to a torch.Tensor (float16 dtype).
    """
    # GeoTIFF - Rasterio
    with rasterio.open(fp=filepath) as dataset:
        array: np.ndarray = dataset.read()
        tensor: torch.Tensor = torch.as_tensor(data=array.astype(dtype="float16"))

    return tensor


class GeoTIFFDataPipeModule(L.LightningDataModule):
    """
    LightningDataModule for loading GeoTIFF files.

    Uses torchdata.
    """

    def __init__(self, batch_size: int = 32):
        """
        Go from datacubes to 256x256 chips!

        Parameters
        ----------
        batch_size : int
            Size of each mini-batch. Default is 32.

        Returns
        -------
        datapipe : torchdata.datapipes.iter.IterDataPipe
            A torch DataPipe that can be passed into a torch DataLoader.
        """
        super().__init__()
        self.batch_size: int = batch_size

    def setup(self, stage: str | None = None):
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        # Step 1 - Get list of GeoTIFF filepaths from data/ folder
        dp_paths = torchdata.datapipes.iter.FileLister(
            root="data/", masks="*.tif", length=1515
        )

        # Step 2 - Split GeoTIFF chips into train/val sets
        # https://pytorch.org/data/0.7/generated/torchdata.datapipes.iter.RandomSplitter.html
        dp_train, dp_val = dp_paths.random_split(
            weights={"train": 0.8, "validation": 0.2}, seed=42
        )

        # Step 3 - Read GeoTIFF into numpy.ndarray, batch and convert to torch.Tensor
        self.datapipe_train = (
            dp_train.map(fn=_array_to_torch).batch(batch_size=self.batch_size).collate()
        )
        self.datapipe_val = (
            dp_val.map(fn=_array_to_torch).batch(batch_size=self.batch_size).collate()
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.datapipe_train,
            batch_size=None,  # handled in datapipe already
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the validation loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.datapipe_val,
            batch_size=None,  # handled in datapipe already
        )
