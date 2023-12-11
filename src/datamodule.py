"""
LightningDataModule to load Earth Observation data from GeoTIFF files using
rasterio.
"""
import pathlib

import lightning as L
import numpy as np
import rasterio
import torch
import torchdata


# %%
def _array_to_torch(filepath: str) -> dict[str, torch.Tensor | str]:
    """
    Read a GeoTIFF file using rasterio into a numpy.ndarray, convert it to a
    torch.Tensor (float16 dtype), and also output spatiotemporal metadata
    associated with the image.

    Parameters
    ----------
    filepath : str
        The path to the GeoTIFF file.

    Returns
    -------
    outputs : dict
        A dictionary containing the following items:
        - image: torch.Tensor - multi-band raster image with shape (Band, Height, Width)
        - bbox: torch.Tensor - spatial bounding box as (xmin, ymin, xmax, ymax)
        - epsg: torch.Tensor - coordinate reference system as an EPSG code
        - date: str - the date the image was acquired in YYYY-MM-DD format
    """
    # GeoTIFF - Rasterio
    with rasterio.open(fp=filepath) as dataset:
        # Get image data
        array: np.ndarray = dataset.read()
        tensor: torch.Tensor = torch.as_tensor(data=array.astype(dtype="float16"))

        # Get spatial bounding box and coordinate reference system in UTM projection
        bbox: torch.Tensor = torch.as_tensor(  # xmin, ymin, xmax, ymax
            data=dataset.bounds, dtype=torch.float64
        )
        epsg: int = torch.as_tensor(data=dataset.crs.to_epsg(), dtype=torch.int32)

        # Get date
        date: str = pathlib.Path(filepath).name[15:25]  # YYYY-MM-DD format

    return {"image": tensor, "bbox": bbox, "epsg": epsg, "date": date}


class GeoTIFFDataPipeModule(L.LightningDataModule):
    """
    LightningDataModule for loading GeoTIFF files.

    Uses torchdata.
    """

    def __init__(
        self,
        data_path: str = "data/",
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        """
        Go from datacubes to 512x512 chips!

        Parameters
        ----------
        data_path : str
            Path to the data folder where the GeoTIFF files are stored. Default
            is 'data/'.
        batch_size : int
            Size of each mini-batch. Default is 32.
        num_workers : int
            How many subprocesses to use for data loading. 0 means that the
            data will be loaded in the main process. Default is 8.

        Returns
        -------
        datapipe : torchdata.datapipes.iter.IterDataPipe
            A torch DataPipe that can be passed into a torch DataLoader.
        """
        super().__init__()
        self.data_path: str = data_path
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

    def setup(self, stage: str | None = None):
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        # Step 1 - Get list of GeoTIFF filepaths from data/ folder
        dp_paths = torchdata.datapipes.iter.FileLister(
            root=self.data_path, masks="*.tif", recursive=True, length=423
        )

        if stage == "fit":  # training/validation loop
            # Step 2 - Split GeoTIFF chips into train/val sets (80%/20%)
            # https://pytorch.org/data/0.7/generated/torchdata.datapipes.iter.RandomSplitter.html
            dp_train, dp_val = dp_paths.random_split(
                weights={"train": 0.8, "validation": 0.2}, total_length=423, seed=42
            )

            # Step 3 - Read GeoTIFF into numpy array, batch and convert to torch.Tensor
            self.datapipe_train = (
                dp_train.map(fn=_array_to_torch)
                .batch(batch_size=self.batch_size)
                .collate()
            )
            self.datapipe_val = (
                dp_val.map(fn=_array_to_torch)
                .batch(batch_size=self.batch_size)
                .collate()
            )

        elif stage == "predict":  # prediction loop
            self.datapipe_predict = (
                dp_paths.map(fn=_array_to_torch)
                .batch(batch_size=self.batch_size)
                .collate()
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.datapipe_train,
            batch_size=None,  # handled in datapipe already
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the validation loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.datapipe_val,
            batch_size=None,  # handled in datapipe already
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the prediction loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.datapipe_predict,
            batch_size=None,  # handled in datapipe already
            num_workers=self.num_workers,
        )
