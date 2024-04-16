from itertools import islice

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
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


class LogDINOPredictions(L.Callback):
    """Visualize the model results at the end of every epoch."""

    def __init__(self):
        """
        Instantiates with wandb-logger.
        """
        super().__init__()
        self.selected_image = None

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        with torch.no_grad():
            # Get WandB logger
            self.logger = get_wandb_logger(trainer=trainer)

            if self.selected_image is None:
                self.selected_image = self.select_image(trainer, pl_module)

            # Update the model weights after batch accumulation and backpropagation
            pl_module.student.load_state_dict(pl_module.teacher.state_dict())

            # if batch_idx % trainer.log_every_n_steps == 0:
            self.log_images(trainer, pl_module)

    def select_image(self, trainer, pl_module):
        print("Selecting image with max variance")
        batches = islice(iter(trainer.val_dataloaders), 2)
        max_variance = -1
        for ibatch in batches:
            batch = {
                k: v.to(pl_module.device)
                for k, v in ibatch.items()
                if isinstance(v, torch.Tensor)
            }
            images = batch["pixels"]  # Shape: [batch_size, channels, height, width]
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

    def log_images(self, trainer, pl_module):
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
        ) = pl_module.student.model.encoder(batch)

        # DECODER
        pixels = pl_module.student.model.decoder(
            encoded_unmasked_patches, unmasked_indices, masked_indices
        )
        pixels = rearrange(
            pixels,
            "b c (h w) (p1 p2) -> b c (h p1) (w p2)",
            h=pl_module.student.model.image_size // pl_module.student.model.patch_size,
            p1=pl_module.student.model.patch_size,
        )

        band_groups = {
            "rgb": (2, 1, 0),
            "<rededge>": (3, 4, 5, 7),
            "<ir>": (6, 8, 9),
            "<sar>": (10, 11),
            "dem": (12,),
        }

        n_rows, n_cols = 3, len(band_groups)
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
                input_img[self.selected_image].detach().cpu().numpy().transpose(1, 2, 0)
            )
            pred_img = (
                pred_img[self.selected_image].detach().cpu().numpy().transpose(1, 2, 0)
            )
            if group_name == "rgb":
                # Normalize RGB images
                input_norm = normalize_img(input_img)
                pred_norm = normalize_img(pred_img)
                # Calculate absolute difference for RGB
                diff_rgb = np.abs(input_norm - pred_norm)
            else:
                # Calculate mean for non-RGB bands if necessary
                dim_imgs = 2
                input_mean = (
                    input_img.mean(axis=dim_imgs)
                    if input_img.ndim > dim_imgs
                    else input_img
                )
                pred_mean = (
                    pred_img.mean(axis=dim_imgs)
                    if pred_img.ndim > dim_imgs
                    else pred_img
                )
                # Normalize and calculate difference
                input_norm = normalize_img(input_mean)
                pred_norm = normalize_img(pred_mean)
                diff_rgb = np.abs(input_norm - pred_norm)

            axs[0, col].imshow(input_norm, cmap="gray" if group_name != "rgb" else None)
            axs[1, col].imshow(pred_norm, cmap="gray" if group_name != "rgb" else None)
            axs[2, col].imshow(diff_rgb, cmap="gray" if group_name != "rgb" else None)

            for ax in axs[:, col]:
                ax.set_title(
                    f"""{group_name} {'Input' if ax == axs[0, col] else
                                            'Pred' if ax == axs[1, col] else
                                            'Diff'}"""
                )
                ax.axis("off")

        plt.tight_layout()
        plt.show()
