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
        for filename in [
            "claytile-12ABC-2022-12-31-01-1",
            "claytile-12ABC-2023-12-31-01-2",
        ]:
            array: np.ndarray = np.ones(shape=(3, 256, 256))
            with rasterio.open(
                fp=f"{tmpdirname}/{filename}.tif",
                mode="w",
                width=256,
                height=256,
                count=3,
                dtype=rasterio.uint16,
                crs="EPSG:32646",
            ) as dst:
                dst.write(array)

        yield tmpdirname


# %%
@pytest.mark.parametrize(
    "stage,dataloader", [("fit", "train_dataloader"), ("predict", "predict_dataloader")]
)
def test_geotiffdatapipemodule(geotiff_folder, stage, dataloader):
    """
    Ensure that GeoTIFFDataPipeModule works to load data from a GeoTIFF file
    into torch.Tensor objects.
    """
    datamodule: L.LightningDataModule = GeoTIFFDataPipeModule(
        data_path=geotiff_folder, batch_size=2
    )

    # Train/validation/predict stage
    datamodule.setup(stage=stage)
    it = iter(getattr(datamodule, dataloader)())
    batch = next(it)

    image = batch["image"]
    bbox = batch["bbox"]
    crs = batch["crs"]
    date = batch["date"]

    assert image.shape == torch.Size([2, 3, 256, 256])
    assert image.dtype == torch.float16

    torch.testing.assert_close(
        actual=bbox,
        expected=torch.tensor(
            data=[[0.0, 256.0, 256.0, 0.0], [0.0, 256.0, 256.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        actual=crs, expected=torch.tensor(data=[32646, 32646], dtype=torch.int32)
    )
    assert date == ["2022-12-31", "2023-12-31"]
