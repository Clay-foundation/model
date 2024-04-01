"""
Tests for custom Callback plugins.

Checks to ensure that the hooks in callbacks can be triggered and produce the
correct output results.
"""

import tempfile

import lightning as L
import pytest
import torch

from src.callbacks_wandb import LogMAEReconstruction
from src.model_vit import ViTLitModule


# %%
def test_callbacks_wandb_log_mae_reconstruction():
    """
    Ensure that the LogMAEReconstruction callback can log a set of images to
    Weights & Biases.
    """
    wandb = pytest.importorskip(modname="wandb")

    # Run tests in a temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        pl_module: L.LightningModule = ViTLitModule()
        trainer: L.Trainer = L.Trainer(
            accelerator="auto",
            callbacks=[LogMAEReconstruction(num_samples=4)],
            devices=1,
            logger=L.pytorch.loggers.WandbLogger(mode="disabled", save_dir=tmpdirname),
            default_root_dir=tmpdirname,
        )
        callback: L.Callback = trainer.callbacks[0]

        assert callback.ready is False  # callback state is not ready at first
        callback.on_sanity_check_end(trainer=trainer, pl_module=pl_module)
        assert callback.ready is True  # callback state is ready after sanity_check

        wandb_images: list[wandb.Image] = callback.on_validation_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs={"logits": torch.rand(4, 64, 53248)},
            batch={"image": torch.rand(4, 13, 512, 512)},
            batch_idx=0,
        )

        # Check that wandb saved some log files to the temporary directory
        # assert os.path.exists(path := f"{tmpdirname}/wandb/latest-run/")
        # assert set(os.listdir(path=path)) == set(
        #     [
        #         f"run-{trainer.logger.version}.wandb",
        #         "tmp",
        #         "files",
        #         "logs",
        #     ]
        # )

        # Check that images logged by WandB have the correct caption and format
        assert len(wandb_images) == 8  # noqa: PLR2004
        assert all(isinstance(w, wandb.Image) for w in wandb_images)
        assert wandb_images[0]._caption == "RGB Image 0"
        assert wandb_images[0].format == "png"
        assert wandb_images[1]._caption == "Reconstructed 0"
        assert wandb_images[1].format == "png"
