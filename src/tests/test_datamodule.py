"""
Tests for GeoTIFFDataPipeModule.

Integration test for the entire data pipeline from loading the data and
pre-processing steps, up to the DataLoader producing mini-batches.
"""
import tempfile

import lightning as L
import numpy as np
import pytest
import rasterio
import torch

from src.datamodule import GeoTIFFDataPipeModule


# %%
@pytest.fixture(scope="function", name="geotiff_folder")
def fixture_geotiff_folder():
    """
    Create a temporary folder containing two GeoTIFF files with random data to
    use in the tests.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename in ["one", "two"]:
            array: np.ndarray = np.ones(shape=(3, 256, 256))
            with rasterio.open(
                fp=f"{tmpdirname}/{filename}.tif",
                mode="w",
                width=256,
                height=256,
                count=3,
                dtype=rasterio.uint16,
            ) as dst:
                dst.write(array)

        yield tmpdirname


# %%
def test_geotiffdatapipemodule(geotiff_folder):
    """
    Ensure that GeoTIFFDataPipeModule works to load data from a GeoTIFF file
    into torch.Tensor objects.
    """
    datamodule: L.LightningDataModule = GeoTIFFDataPipeModule(
        data_path=geotiff_folder, batch_size=2
    )
    datamodule.setup()

    it = iter(datamodule.train_dataloader())
    image = next(it)

    assert image.shape == torch.Size([2, 3, 256, 256])
    assert image.dtype == torch.float16
