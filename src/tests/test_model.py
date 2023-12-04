"""
Tests for neural network model architecture.

Based loosely on Lightning's testing method described at
https://github.com/Lightning-AI/lightning/blob/2.1.0/.github/CONTRIBUTING.md#how-to-add-new-tests
"""
import lightning as L
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
            {"image": torch.randn(2, 13, 256, 256).to(dtype=torch.float16)},
            {"image": torch.randn(2, 13, 256, 256).to(dtype=torch.float16)},
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

    # Training
    trainer: L.Trainer = L.Trainer(
        accelerator="auto", devices=1, precision="16-mixed", fast_dev_run=True
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Test/Evaluation
    # TODO
