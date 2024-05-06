"""
LightningDataModule to load Earth Observation data from GeoTIFF files using
rasterio.
"""

import math
from collections import defaultdict
from pathlib import Path
from typing import List, Literal

import lightning as L
import numpy as np
import rasterio
import torch
import torchdata
import yaml
from box import Box
from einops import rearrange
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import v2


# %%
# Regular torch Dataset
class ClayDataset(Dataset):
    def __init__(self, chips_path: List[Path], transform=None):
        super().__init__()
        self.chips_path = chips_path
        self.transform = transform

    def normalize_timestamp(self, ts):
        year, month, day = map(np.float32, ts.split("-"))
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

    def normalize_latlon(self, lon, lat):
        lon_radians = math.radians(lon)
        lat_radians = math.radians(lat)

        # Apply sine and cosine
        lon = math.atan2(
            math.cos(lon_radians),
            math.sin(lon_radians),
        )
        lat = math.sin(lat_radians)
        return lon, lat

    def read_chip(self, chip_path):
        chip = rasterio.open(chip_path)

        # read timestep & normalize
        date = chip.tags()["date"]  # YYYY-MM-DD
        year, month, day = self.normalize_timestamp(date)

        # read lat,lon from UTM to WGS84 & normalize
        bounds = chip.bounds  # xmin, ymin, xmax, ymax
        epsg = chip.crs.to_epsg()  # e.g. 32632
        lon, lat = chip.lnglat()  # longitude, latitude
        lon, lat = self.normalize_latlon(lon, lat)

        return {
            "pixels": chip.read(),
            # Raw values
            "bbox": bounds,
            "epsg": epsg,
            "date": date,
            # Normalized values
            "latlon": (lat, lon),
            "timestep": (year, month, day),
        }

    def __getitem__(self, idx):
        chip_path = self.chips_path[idx]
        cube = self.read_chip(chip_path)

        # remove nans and convert to tensor
        cube["pixels"] = torch.as_tensor(data=cube["pixels"], dtype=torch.float32)
        cube["bbox"] = torch.as_tensor(data=cube["bbox"], dtype=torch.float64)
        cube["epsg"] = torch.as_tensor(data=cube["epsg"], dtype=torch.int32)
        cube["date"] = str(cube["date"])
        cube["latlon"] = torch.as_tensor(data=cube["latlon"])
        cube["timestep"] = torch.as_tensor(data=cube["timestep"])
        try:
            cube["source_url"] = str(chip_path.absolute())
        except AttributeError:
            cube["source_url"] = chip_path

        if self.transform:
            # Normalize data
            cube["pixels"] = self.transform(cube["pixels"])

        return cube

    def __len__(self):
        return len(self.chips_path)


class EODataset(Dataset):
    """Reads different Earth Observation data sources from a directory."""

    def __init__(self, chips_path: List[Path], size: int, metadata: Box) -> None:
        super().__init__()
        self.chips_path = chips_path
        self.size = size
        self.transforms = {}

        # Platforms to setup transforms for
        platforms = [
            "landsat-c2l1",
            "landsat-c2l2-sr",
            "linz",
            "naip",
            "sentinel-1-rtc",
            "sentinel-2-l2a",
        ]

        # Generate transforms for each platform using a helper function
        for platform in platforms:
            mean = list(metadata[platform].bands.mean.values())
            std = list(metadata[platform].bands.std.values())
            self.transforms[platform] = self.create_transforms(mean, std)

    def create_transforms(self, mean, std):
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomCrop(size=(self.size, self.size)),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.chips_path)

    def __getitem__(self, idx):
        chip_path = self.chips_path[idx]
        with np.load(chip_path, allow_pickle=False) as chip:
            pixels = torch.from_numpy(chip["pixels"].astype(np.float32))
            platform = chip_path.parent.name
            pixels = self.transforms[platform](pixels)

            # Prepare additional information
            additional_info = {
                "platform": platform,
                "time": torch.tensor(
                    np.hstack((chip["week_norm"], chip["hour_norm"])),
                    dtype=torch.float32,
                ),
                "latlon": torch.tensor(
                    np.hstack((chip["lat_norm"], chip["lon_norm"])), dtype=torch.float32
                ),
            }

        return {"pixels": pixels, **additional_info}


class ClaySampler(Sampler):
    def __init__(self, dataset, platforms, batch_size):
        self.dataset = dataset
        self.platforms = platforms
        self.batch_size = batch_size

        self.cubes_per_platform = {platform: [] for platform in platforms}
        for idx, chip_path in enumerate(self.dataset.chips_path):
            platform = chip_path.parent.name
            self.cubes_per_platform[platform].append(idx)

    def __iter__(self):
        cubes_per_platform_per_epoch = {}
        rng = np.random.default_rng()
        # Shuffle and adjust sizes
        max_len = max(len(indices) for indices in self.cubes_per_platform.values())
        for platform in self.platforms:
            indices = self.cubes_per_platform[platform]
            rng.shuffle(indices)
            repeated_indices = np.tile(indices, (max_len // len(indices) + 1))[:max_len]
            cubes_per_platform_per_epoch[platform] = repeated_indices

        # Create batches such that we return one platform per batch in cycle
        # Ignore the last batch if it is incomplete
        for i in range(0, max_len, self.batch_size):
            for platform in self.platforms:
                batch = cubes_per_platform_per_epoch[platform][i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch

    def __len__(self):
        return len(self.dataset.chips_path) // self.batch_size


def batch_collate(batch):
    """Collate function for DataLoader.

    Merge the first two dimensions of the input tensors.
    """
    d = defaultdict(list)
    for item in batch:
        d["pixels"].append(item["pixels"])
        d["time"].append(item["time"])
        d["latlon"].append(item["latlon"])
        d["platform"].append(item["platform"])
    return {
        "pixels": rearrange(d["pixels"], "b1 b2 c h w -> (b1 b2) c h w"),
        "time": rearrange(d["time"], "b1 b2 t -> (b1 b2) t"),
        "latlon": rearrange(d["latlon"], "b1 b2 ll -> (b1 b2) ll"),
        "platform": d["platform"],
    }


class ClayDataModule(L.LightningDataModule):
    def __init__(  # noqa: PLR0913
        self,
        data_dir: str = "data",
        size: int = 224,
        metadata_path: str = "configs/metadata.yaml",
        platforms: list = [
            "landsat-c2l1",
            "landsat-c2l2-sr",
            "linz",
            "naip",
            "sentinel-1-rtc",
            "sentinel-2-l2a",
        ],
        batch_size: int = 10,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.size = size
        self.platforms = platforms
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = 0.8

    def setup(self, stage: Literal["fit", "predict"] | None = None) -> None:
        # Get list of GeoTIFF filepaths from s3 bucket or data/ folder
        if self.data_dir.startswith("s3://"):
            dp = torchdata.datapipes.iter.IterableWrapper(iterable=[self.data_dir])
            chips_path = list(dp.list_files_by_s3(masks="*.npz"))
        else:  # if self.data_dir is a local data path
            chips_path = sorted(list(Path(self.data_dir).glob("**/*.npz")))
            chips_platform = [chip.parent.parent.name for chip in chips_path]
            # chips_platform = [chip.parent.parent.name for chip in chips_path]
        print(f"Total number of chips: {len(chips_path)}")

        if stage == "fit":
            trn_paths, val_paths = train_test_split(
                chips_path,
                test_size=(1 - self.split_ratio),
                stratify=chips_platform,
                shuffle=True,
            )

            self.trn_ds = EODataset(
                chips_path=trn_paths,
                size=self.size,
                metadata=self.metadata,
            )
            self.trn_sampler = ClaySampler(
                dataset=self.trn_ds,
                platforms=self.platforms,
                batch_size=self.batch_size,
            )
            self.val_ds = EODataset(
                chips_path=val_paths,
                size=self.size,
                metadata=self.metadata,
            )
            self.val_sampler = ClaySampler(
                dataset=self.val_ds,
                platforms=self.platforms,
                batch_size=self.batch_size,
            )

        elif stage == "predict":
            self.prd_ds = EODataset(
                chips_path=chips_path,
                platform=self.platform,
                metadata_path=self.metadata_path,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            num_workers=self.num_workers,
            batch_sampler=self.trn_sampler,
            collate_fn=batch_collate,
            pin_memory=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.num_workers,
            batch_sampler=self.val_sampler,
            collate_fn=batch_collate,
            pin_memory=True,
            prefetch_factor=4,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.prd_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


# %%
# Torchdata-based approach
def _array_to_torch(filepath: str) -> dict[str, torch.Tensor | str]:
    """
    Read a GeoTIFF file using rasterio into a numpy.ndarray, convert it to a
    torch.Tensor (float32 dtype), and also output spatiotemporal metadata
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
        - source_url: str - the URL or path to the source GeoTIFF file
    """
    # GeoTIFF - Rasterio
    with rasterio.open(fp=filepath) as dataset:
        # Get image data
        array: np.ndarray = dataset.read()
        tensor: torch.Tensor = torch.as_tensor(data=array.astype(dtype="float32"))

        # Get spatial bounding box and coordinate reference system in UTM projection
        bbox: torch.Tensor = torch.as_tensor(  # xmin, ymin, xmax, ymax
            data=dataset.bounds, dtype=torch.float64
        )
        epsg: int = torch.as_tensor(data=dataset.crs.to_epsg(), dtype=torch.int32)

        # Get date
        date: str = dataset.tags()["date"]  # YYYY-MM-DD format

    return {
        "image": tensor,  # shape (13, 512, 512)
        "bbox": bbox,  # bounds [xmin, ymin, xmax, ymax]
        "epsg": epsg,  # e.g. 32632
        "date": date,  # e.g. 2020-12-31
        "source_url": filepath,  # e.g. s3://.../claytile_12ABC_20201231_v0_0200.tif
    }


class GeoTIFFDataPipeModule(L.LightningDataModule):
    """
    LightningDataModule for loading GeoTIFF files.

    Uses torchdata.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        """
        Go from datacubes to 512x512 chips!

        Parameters
        ----------
        data_dir : str
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
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

    def setup(self, stage: Literal["fit", "predict"] | None = None):
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.

        Parameters
        ----------
        stage : str or None
            Whether to setup the datapipe for the training/validation loop, or
            the prediction loop. Choose from either 'fit' or 'predict'.
        """
        # Step 1 - Get list of GeoTIFF filepaths from s3 bucket or data/ folder
        if self.data_dir.startswith("s3://"):
            dp = torchdata.datapipes.iter.IterableWrapper(iterable=[self.data_dir])
            self.dp_paths = dp.list_files_by_s3(masks="*.tif")
        else:  # if self.data_dir is a local data path
            self.dp_paths = torchdata.datapipes.iter.FileLister(
                root=self.data_dir, masks="*.tif", recursive=True
            )

        if stage == "fit":  # training/validation loop
            # Step 2 - Split GeoTIFF chips into train/val sets (80%/20%)
            # https://pytorch.org/data/0.7/generated/torchdata.datapipes.iter.RandomSplitter.html
            dp_train, dp_val = self.dp_paths.random_split(
                weights={"train": 0.8, "validation": 0.2}, total_length=423, seed=42
            )

            # Step 3 - Read GeoTIFF into numpy array, batch and convert to torch.Tensor
            self.datapipe_train = (
                dp_train.sharding_filter()
                .map(fn=_array_to_torch)
                .batch(batch_size=self.batch_size)
                .collate()
            )
            self.datapipe_val = (
                dp_val.sharding_filter()
                .map(fn=_array_to_torch)
                .batch(batch_size=self.batch_size)
                .collate()
            )

        elif stage == "predict":  # prediction loop
            self.datapipe_predict = (
                self.dp_paths.sharding_filter()
                .map(fn=_array_to_torch)
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
