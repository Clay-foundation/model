import lightning as L
import matplotlib.pyplot as plt
import torch
import wandb
from einops import rearrange
from lightning.pytorch.callbacks import Callback


class LogIntermediatePredictions(Callback):
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
        At the end of each epoch, takes the first batch from validation dataset &
        logs the model predictions to wandb-logger for humans to interpret how model evolves over time.
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
                pixels, "b c (h w) (p1 p2) -> b c (h p1) (w p2)", h=8, p1=32
            )

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
