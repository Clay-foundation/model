"""
Lightning callback for logging intermediate predictions to Weights & Biases.

References:
- https://lightning.ai/docs/pytorch/2.1.0/common/trainer.html#callbacks
"""

from typing import Any, cast

import lightning as L
import lightning.pytorch.loggers as pl_loggers
import matplotlib.pyplot as plt
import torch
from einops import rearrange

try:
    import wandb
except ImportError:
    wandb = None


def get_wandb_logger(trainer: L.Trainer) -> pl_loggers.WandbLogger:
    """Safely get Weights & Biases logger from Trainer."""
    if getattr(trainer, "fast_dev_run", False):
        raise RuntimeError(
            "Cannot use wandb callbacks since pytorch lightning disables "
            "loggers in `fast_dev_run=true` mode."
        )

    for logger in trainer.loggers:
        if isinstance(logger, pl_loggers.WandbLogger):
            return logger

    raise RuntimeError(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class LogIntermediatePredictions(L.Callback):
    """Visualize the model results at the end of every epoch."""

    def __init__(self) -> None:
        """
        Instantiates with wandb-logger.
        """
        super().__init__()

    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """
        Called when the validation loop ends.
        At the end of each epoch, takes the first batch from validation dataset
        & logs the model predictions to wandb-logger for humans to interpret
        how model evolves over time.
        """
        with torch.no_grad():
            if wandb is None:
                raise RuntimeError("wandb is required for LogIntermediatePredictions.")

            # Get WandB logger
            self.logger = get_wandb_logger(trainer=trainer)

            # get the val dataloader
            val_loaders = trainer.val_dataloaders
            if val_loaders is None:
                return
            val_dl = iter(val_loaders)
            datamodule = getattr(trainer, "datamodule")  # noqa: B009
            clay_module = cast("Any", pl_module)
            for _i in range(6):
                batch = next(val_dl)
                platform = batch["platform"][0]

                batch = {
                    k: v.to(pl_module.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                waves = torch.tensor(list(datamodule.metadata[platform].bands.wavelength.values()))
                gsd = torch.tensor(datamodule.metadata[platform].gsd)

                # ENCODER
                (
                    encoded_unmasked_patches,
                    unmasked_indices,
                    masked_indices,
                    masked_matrix,
                ) = clay_module.model.encoder(
                    {
                        "pixels": batch["pixels"],
                        "time": batch["time"],
                        "latlon": batch["latlon"],
                        "gsd": gsd,
                        "waves": waves,
                    }
                )

                # DECODER
                pixels, waves = clay_module.model.decoder(
                    encoded_unmasked_patches,
                    unmasked_indices,
                    masked_indices,
                    masked_matrix,
                    batch["time"],
                    batch["latlon"],
                    gsd,
                    waves,
                )  # pixels: batch x (patch x patch) x 1024
                pixels = rearrange(
                    pixels,
                    "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                    p1=clay_module.model.patch_size,
                    p2=clay_module.model.patch_size,
                    h=datamodule.size // clay_module.model.patch_size,
                    w=datamodule.size // clay_module.model.patch_size,
                )

                assert pixels.shape == batch["pixels"].shape
                batch["pixels"] = batch["pixels"].detach().cpu().numpy()
                pixels = pixels.detach().cpu().numpy()

                n_rows = 4  # 2 for actual and 2 for predicted
                n_cols = 8

                fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 8))

                for j in range(n_cols):
                    # Plot actual images in rows 0 and 2
                    axs[0, j].imshow(batch["pixels"][j][0], cmap="viridis")
                    axs[0, j].set_title(f"Actual {j}")
                    axs[0, j].axis("off")

                    axs[2, j].imshow(
                        batch["pixels"][j + n_cols][0],
                        cmap="viridis",
                    )
                    axs[2, j].set_title(f"Actual {j + n_cols}")
                    axs[2, j].axis("off")

                    # Plot predicted images in rows 1 and 3
                    axs[1, j].imshow(pixels[j][0], cmap="viridis")
                    axs[1, j].set_title(f"Pred {j}")
                    axs[1, j].axis("off")

                    axs[3, j].imshow(pixels[j + n_cols][0], cmap="viridis")
                    axs[3, j].set_title(f"Pred {j + n_cols}")
                    axs[3, j].axis("off")

                self.logger.experiment.log({f"{platform}": wandb.Image(fig)})
            plt.close(fig)
