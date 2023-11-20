"""
Tests for neural network model architecture.

Based loosely on Lightning's testing method described at
https://github.com/Lightning-AI/lightning/blob/2.1.0/.github/CONTRIBUTING.md#how-to-add-new-tests
"""
import lightning as L

from src.datamodule import BaseDataModule
from src.model import MAELitModule


# %%
def test_model():
    """
    Run a full train, val, test and prediction loop using 1 batch.
    """
    # Get some random data
    dataloader: L.LightningDataModule = BaseDataModule()

    # Initialize model
    model: L.LightningModule = MAELitModule()

    # Training
    trainer: L.Trainer = L.Trainer(accelerator="auto", devices=1, fast_dev_run=True)
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Test/Evaluation
    # TODO
