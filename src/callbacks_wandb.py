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
from itertools import islice

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
def get_wandb_logger(trainer: L.Trainer) -> L.pytorch.loggers.WandbLogger:
    """
    Safely get Weights & Biases logger from Trainer.
    """

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables "
            "loggers in `fast_dev_run=true` mode."
        )

    for logger in trainer.loggers:
        if isinstance(logger, L.pytorch.loggers.WandbLogger):
            return logger
            break

    raise Exception(
        "You are using wandb related callback, "
        "but WandbLogger was not found for some reason..."
    )


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
                self.logger = get_wandb_logger(trainer=trainer)

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
                self.logger.experiment.log(data={"Examples": figures})

            return figures


class LogIntermediatePredictions(L.Callback):
    """Visualize the model results at the end of every epoch."""

    def __init__(self):
        """
        Instantiates with wandb-logger.
        """
        super().__init__()
        self.selected_image = None

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
            # Get WandB logger
            self.logger = get_wandb_logger(trainer=trainer)

            if self.selected_image is None:
                self.selected_image = self.select_image(trainer,pl_module)
            self.log_images(trainer,pl_module)


    def select_image(self,trainer,pl_module):
        print("Selecting image with max variance")
        batches = islice(iter(trainer.val_dataloaders), 3)
        max_variance = -1
        for ibatch in batches:
            batch = {
                k: v.to(pl_module.device)
                for k, v in ibatch.items()
                if isinstance(v, torch.Tensor)
            }
            images = batch[
                "pixels"
            ]  # Shape: [batch_size, channels, height, width]
            variances = images.var(
                dim=[1, 2, 3], keepdim=False
            )  # Calculate variance across C, H, W dimensions
            max_var_index = torch.argmax(variances).item()
            if variances[max_var_index] > max_variance:
                max_variance = variances[max_var_index]
                self.selected_image = max_var_index
        assert self.selected_image is not None
        print(f"Selected image with max variance: {self.selected_image}")
        return self.selected_image

    def log_images(self,trainer,pl_module):
        if self.selected_image >= trainer.val_dataloaders.batch_size:
            batch = next(
                islice(
                    iter(trainer.val_dataloaders),
                    self.selected_image // trainer.val_dataloaders.batch_size,
                    None,
                )
            )
        else:
            batch = next(iter(trainer.val_dataloaders))

        batch = {
            k: v.to(pl_module.device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
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

        band_groups = {
            "rgb": (2, 1, 0),
            "<rededge>": (3, 4, 5, 7),
            "<ir>": (6, 8, 9),
            "<sar>": (10, 11),
            "dem": (12,),
        }

        n_rows, n_cols = (
            3,
            len(band_groups),
        )  # Rows for Input, Prediction, Difference
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

        def normalize_img(img):
            lower_percentile, upper_percentile = 1, 99
            lower_bound = np.percentile(img, lower_percentile)
            upper_bound = np.percentile(img, upper_percentile)
            img_clipped = np.clip(img, lower_bound, upper_bound)
            return (img_clipped - img_clipped.min()) / (
                img_clipped.max() - img_clipped.min()
            )

        for col, (group_name, bands) in enumerate(band_groups.items()):
            input_img = batch["pixels"][:, bands, :, :]
            pred_img = pixels[:, bands, :, :]
            input_img = (
                input_img[self.selected_image]
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            pred_img = (
                pred_img[self.selected_image]
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )

            if group_name == "rgb":
                # Normalize RGB images
                input_norm = normalize_img(input_img)
                pred_norm = normalize_img(pred_img)
                # Calculate absolute difference for RGB
                diff_rgb = np.abs(input_norm - pred_norm)
            else:
                # Calculate mean for non-RGB bands if necessary
                input_mean = (
                    input_img.mean(axis=2) if input_img.ndim > 2 else input_img  # noqa: PLR2004
                )
                pred_mean = pred_img.mean(axis=2) if pred_img.ndim > 2 else pred_img  # noqa: PLR2004
                # Normalize and calculate difference
                input_norm = normalize_img(input_mean)
                pred_norm = normalize_img(pred_mean)
                diff_rgb = np.abs(input_norm - pred_norm)

            axs[0, col].imshow(
                input_norm, cmap="gray" if group_name != "rgb" else None
            )
            axs[1, col].imshow(
                pred_norm, cmap="gray" if group_name != "rgb" else None
            )
            axs[2, col].imshow(
                diff_rgb, cmap="gray" if group_name != "rgb" else None
            )

            for ax in axs[:, col]:
                ax.set_title(
                    f"""{group_name} {'Input' if ax == axs[0, col] else
                                     'Pred' if ax == axs[1, col] else
                                     'Diff'}"""
                )
                ax.axis("off")

        plt.tight_layout()
        self.logger.experiment.log({"Images": wandb.Image(fig)})
        plt.close(fig)


# %%
