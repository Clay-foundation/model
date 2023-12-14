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
import numpy as np
import skimage
import torch
import wandb


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
    ):
        """
        Called in the validation loop at the start of every mini-batch.

        Gather data from a single batch.
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
                # assert y_hat.shape == torch.Size([6, 512, 512, 13])
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
