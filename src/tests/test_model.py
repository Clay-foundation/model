"""
Tests for neural network model architecture.

Based loosely on Lightning's testing method described at
https://github.com/Lightning-AI/lightning/blob/2.1.0/.github/CONTRIBUTING.md#how-to-add-new-tests
"""

import os
import tempfile

import geopandas as gpd
import lightning as L
import numpy as np
import pytest
import torch
import torchdata
import torchdata.dataloader2

from src.model_clay import CLAYModule
from src.model_vit import ViTLitModule


# %%
@pytest.fixture(scope="function", name="datapipe")
def fixture_datapipe() -> torchdata.datapipes.iter.IterDataPipe:
    """
    A torchdata DataPipe with random data to use in the tests.
    """
    datapipe = torchdata.datapipes.iter.IterableWrapper(
        iterable=[
            {
                # For ViTLitModule
                "image": torch.randn(3, 13, 512, 512).to(dtype=torch.float32),
                # For CLAYModule
                "pixels": torch.randn(3, 13, 512, 512).to(dtype=torch.float32),
                "timestep": torch.tensor(
                    data=[(2020, 1, 1), (2021, 6, 15), (2022, 12, 31)],
                    dtype=torch.float32,
                ),
                "latlon": torch.tensor(
                    data=[(12, 34), (56, 78), (90, 100)], dtype=torch.float32
                ),
                # For both
                "bbox": torch.tensor(
                    data=[
                        [499975.0, 3397465.0, 502535.0, 3400025.0],
                        [530695.0, 3397465.0, 533255.0, 3400025.0],
                        [561415.0, 3397465.0, 563975.0, 3400025.0],
                    ]
                ),
                "date": ["2020-01-01", "2021-06-15", "2022-12-31"],
                "epsg": torch.tensor(data=[32760, 32760, 32760]),
                "source_url": [
                    "s3://claytile_60HTE_1.tif",
                    "s3://claytile_60GUV_2.tif",
                    "s3://claytile_60GUV_3.tif",
                ],
            },
        ]
    )
    return datapipe


# %%
def test_model_vit_fit(datapipe):
    """
    Run a full train and validation loop using 1 batch.
    """
    # Get some random data
    dataloader = torchdata.dataloader2.DataLoader2(datapipe=datapipe)

    # Initialize model
    model: L.LightningModule = ViTLitModule()

    # Run tests in a temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Training
        trainer: L.Trainer = L.Trainer(
            accelerator=(
                "cpu"  # fallback to CPU on osx-arm64 CI
                if os.getenv("PYTORCH_MPS_PREFER_METAL") == "0"
                and torch.backends.mps.is_available()
                else "auto"
            ),
            devices=1,
            precision="16-mixed",
            fast_dev_run=True,
            default_root_dir=tmpdirname,
        )
        trainer.fit(model=model, train_dataloaders=dataloader)


@pytest.mark.parametrize(
    "litmodule,precision",
    [
        (CLAYModule, "16-mixed" if torch.cuda.is_available() else "32-true"),
        (ViTLitModule, "16-mixed"),
    ],
)
@pytest.mark.parametrize("embeddings_level", ["mean", "patch", "group"])
def test_model_predict(datapipe, litmodule, precision, embeddings_level):
    """
    Run a single prediction loop using 1 batch.
    """
    # Get some random data
    dataloader = torchdata.dataloader2.DataLoader2(datapipe=datapipe)

    # Initialize model
    if litmodule == CLAYModule:
        litargs = {
            "embeddings_level": embeddings_level,
        }
    else:
        litargs = {}

    model: L.LightningModule = litmodule(**litargs)

    # Run tests in a temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Training
        trainer: L.Trainer = L.Trainer(
            accelerator=(
                "cpu"  # fallback to CPU on osx-arm64 CI
                if os.getenv("PYTORCH_MPS_PREFER_METAL") == "0"
                and torch.backends.mps.is_available()
                else "auto"
            ),
            devices="auto",
            precision=precision,
            fast_dev_run=True,
            default_root_dir=tmpdirname,
        )

        # Prediction
        trainer.predict(model=model, dataloaders=dataloader)
        assert (
            len(os.listdir(path=f"{tmpdirname}/data/embeddings")) == 2  # noqa: PLR2004
        )
        assert os.path.exists(
            path := f"{tmpdirname}/data/embeddings/60HTE_20200101_20200101_v001.gpq"
        )
        assert os.path.exists(
            path := f"{tmpdirname}/data/embeddings/60GUV_20210615_20221231_v001.gpq"
        )
        geodataframe: gpd.GeoDataFrame = gpd.read_parquet(path=path)

        assert geodataframe.shape == (2, 5)  # 2 rows, 5 columns
        assert list(geodataframe.columns) == [
            "index",
            "source_url",
            "date",
            "embeddings",
            "geometry",
        ]
        assert geodataframe.index.dtype == "int64"
        assert geodataframe.source_url.dtype == "string"
        assert geodataframe.date.dtype == "date32[day][pyarrow]"
        assert geodataframe.embeddings.dtype == "object"
        assert geodataframe.geometry.dtype == gpd.array.GeometryDtype()

        expected_shape_lookup = {
            "mean": (768,),
            "patch": (16 * 16 * 768,),
            "group": (6 * 16 * 16 * 768,),
        }

        for embeddings in geodataframe.embeddings:
            assert (
                embeddings.shape == expected_shape_lookup[embeddings_level]
                if litmodule == CLAYModule
                else (768,)
            )
            assert embeddings.dtype == "float32"
            assert not np.isnan(embeddings).any()
