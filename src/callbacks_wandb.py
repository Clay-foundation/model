"""
Lightning callback functions for logging to Weights & Biases.

Includes a way to visualize RGB images derived from the raw logits of a Masked
Autoencoder's decoder during the validation loop. I.e. to see if the Vision
Transformer model is learning how to do image reconstruction.

Usage:

```
import lightning as L

from src.callbacks_wandb import LogMAEReconstruction

trainer = L.Trainer(
    ...,
    callbacks=[LogMAEReconstruction(num_samples=6)]
)
```

References:
- https://lightning.ai/docs/pytorch/2.1.0/common/trainer.html#callbacks
- https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
- https://github.com/ashleve/lightning-hydra-template/blob/wandb-callbacks/src/callbacks/wandb_callbacks.py#L245
"""
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from einops import rearrange

try:
    import wandb
except ImportError:
    wandb = None


# %%
class LogMAEReconstruction(L.Callback):
    """
    Logs reconstructed RGB images from a Masked Autoencoder's decoder to WandB.
    """

    def __init__(self, num_samples: int = 8):
        """
        Define how many sample images to log.

        Parameters
        ----------
        num_samples : int
            The number of RGB image samples to upload to WandB. Default is 8.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.ready: bool = False

        if wandb is None:
            raise ModuleNotFoundError(
                "Package `wandb` is required to be installed to use this callback. "
                "Please use `pip install wandb` or "
                "`conda install -c conda-forge wandb` "
                "to install the package"
            )

    def on_sanity_check_start(self, trainer, pl_module):
        """
        Don't execute callback before validation sanity checks are completed.
        """
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """
        Start executing callback only after all validation sanity checks end.
        """
        self.ready = True

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor | list[str]],
        batch_idx: int,
    ) -> list:
        """
        Called in the validation loop at the start of every mini-batch.

        Gather a sample of data from the first mini-batch, get the RGB bands,
        apply histogram equalization to the image, and log it to WandB.
        """
        if self.ready and batch_idx == 0:  # only run on first mini-batch
            with torch.inference_mode():
                # Get WandB logger
                for logger in trainer.loggers:
                    if isinstance(logger, L.pytorch.loggers.WandbLogger):
                        break

                # Turn raw logits into reconstructed 512x512 images
                patchified_pixel_values: torch.Tensor = outputs["logits"]
                # assert patchified_pixel_values.shape == torch.Size([32, 64, 53248])
                y_hat: torch.Tensor = pl_module.vit.unpatchify(
                    patchified_pixel_values=patchified_pixel_values
                )
                # assert y_hat.shape == torch.Size([32, 13, 512, 512])

                # Reshape tensors from channel-first to channel-last
                x: torch.Tensor = torch.einsum(
                    "bchw->bhwc", batch["image"][: self.num_samples]
                )
                y_hat: torch.Tensor = torch.einsum(
                    "bchw->bhwc", y_hat[: self.num_samples]
                )
                # assert y_hat.shape == torch.Size([8, 512, 512, 13])
                assert x.shape == y_hat.shape

                # Plot original and reconstructed RGB images of Sentinel-2
                rgb_original: np.ndarray = (
                    x[:, :, :, [2, 1, 0]].cpu().to(dtype=torch.float32).numpy()
                )
                rgb_reconstruction: np.ndarray = (
                    y_hat[:, :, :, [2, 1, 0]].cpu().to(dtype=torch.float32).numpy()
                )

                figures: list[wandb.Image] = []
                for i in range(min(x.shape[0], self.num_samples)):
                    img_original = wandb.Image(
                        data_or_path=skimage.exposure.equalize_hist(
                            image=rgb_original[i]
                        ),
                        caption=f"RGB Image {i}",
                    )
                    figures.append(img_original)

                    img_reconstruction = wandb.Image(
                        data_or_path=skimage.exposure.equalize_hist(
                            image=rgb_reconstruction[i]
                        ),
                        caption=f"Reconstructed {i}",
                    )
                    figures.append(img_reconstruction)

                # Upload figures to WandB
                logger.experiment.log(data={"Examples": figures})

            return figures


class LogIntermediatePredictions(L.Callback):
    """Visualize the model results at the end of every epoch."""

    def __init__(self, logger):
        """Instantiates with wandb-logger.
        Args:
            logger : wandb-logger instance.
        """
        super().__init__()
        self.logger = logger

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
            # get the first batch from trainer
            batch = next(iter(trainer.val_dataloaders))
            batch = {k: v.to(pl_module.device) for k, v in batch.items()}
            # ENCODER
            (
                encoded_unmasked_patches,
                unmasked_indices,
                masked_indices,
                masked_matrix,
            ) = pl_module.model.encoder(batch)

            # DECODER
            pixels = pl_module.model.decoder(
                encoded_unmasked_patches, unmasked_indices, masked_indices
            )
            pixels = rearrange(
                pixels,
                "b c (h w) (p1 p2) -> b c (h p1) (w p2)",
                h=pl_module.model.image_size // pl_module.model.patch_size,
                p1=pl_module.model.patch_size,
            )

            assert pixels.shape == batch["pixels"].shape

            n_rows = 2
            n_cols = 8

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 4))

            for i in range(n_cols):
                axs[0, i].imshow(
                    batch["pixels"][i][0].detach().cpu().numpy(), cmap="bwr"
                )
                axs[0, i].set_title(f"Image {i}")
                axs[0, i].axis("off")

                axs[1, i].imshow(pixels[i][0].detach().cpu().numpy(), cmap="gray")
                axs[1, i].set_title(f"Preds {i}")
                axs[1, i].axis("off")

            wandb.log({"Images": wandb.Image(fig)})
