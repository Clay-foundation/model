"""
Tests for neural network model architecture.

Based loosely on Lightning's testing method described at
https://github.com/Lightning-AI/lightning/blob/2.1.0/.github/CONTRIBUTING.md#how-to-add-new-tests
"""
import os
import tempfile

import lightning as L
import numpy as np
import pytest
import torch
import torchdata
import torchdata.dataloader2

from src.model_vit import ViTLitModule


# %%
@pytest.fixture(scope="function", name="datapipe")
def fixture_datapipe() -> torchdata.datapipes.iter.IterDataPipe:
    """
    A torchdata DataPipe with random data to use in the tests.
    """
    datapipe = torchdata.datapipes.iter.IterableWrapper(
        iterable=[
            torch.randn(2, 13, 256, 256).to(dtype=torch.float16),
            torch.randn(2, 13, 256, 256).to(dtype=torch.float16),
        ]
    )
    return datapipe


# %%
def test_model_vit(datapipe):
    """
    Run a full train, val, test and prediction loop using 1 batch.
    """
    # Get some random data
    dataloader = torchdata.dataloader2.DataLoader2(datapipe=datapipe)

    # Initialize model
    model: L.LightningModule = ViTLitModule()

    # Run tests in a temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Training
        trainer: L.Trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision="16-mixed",
            fast_dev_run=True,
            default_root_dir=tmpdirname,
        )
        trainer.fit(model=model, train_dataloaders=dataloader)

        # Prediction
        trainer.predict(model=model, dataloaders=dataloader)
        assert os.path.exists(path := f"{tmpdirname}/data/embeddings/embedding_0.npy")
        embedding: np.ndarray = np.load(file=path)
        assert embedding.shape == (2, 17, 768)
        assert embedding.dtype == "float32"
