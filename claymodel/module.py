__all__ = ["ClayMAEModule"]

from typing import Literal

import lightning as L
import torch

from claymodel.metadata import PlatformMetadata, load_metadata_yaml
from claymodel.model import (
    ClayMAE,
    Encoder,
    clay_mae_base,
    clay_mae_large,
    clay_mae_small,
    clay_mae_tiny,
)


class ClayMAEModule(L.LightningModule):
    model: ClayMAE
    metadata: dict[str, PlatformMetadata]

    def __init__(  # noqa: PLR0913
        self,
        model_size: str = "base",
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = False,
        patch_size: int = 8,
        shuffle: bool = False,
        metadata_path: str = "claymodel/configs/metadata.yaml",
        teacher: str = "vit_large_patch14_reg4_dinov2.lvd142m",
        dolls: list[int] = [16, 32, 64, 128, 256, 768],
        doll_weights: list[float] = [1, 1, 1, 1, 1, 1],
        matryoshka: bool = False,
        lr: float = 1e-5,
        wd: float = 0.05,
        b1: float = 0.9,
        b2: float = 0.95,
        embeddings_level: Literal["mean", "patch", "group"] = "mean",  # noqa: E501 unused, kept for ckpt compat
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.metadata = load_metadata_yaml(metadata_path)
        model_map = {
            "tiny": clay_mae_tiny,
            "small": clay_mae_small,
            "base": clay_mae_base,
            "large": clay_mae_large,
        }
        if model_size in model_map:
            model_args = {
                "mask_ratio": mask_ratio,
                "patch_size": patch_size,
                "norm_pix_loss": norm_pix_loss,
                "shuffle": shuffle,
                "metadata": self.metadata,
                "teacher": teacher,
                "dolls": dolls,
                "doll_weights": doll_weights,
                "matryoshka": matryoshka,
            }
            self.model = model_map[model_size](**model_args)
            # NOTE: Weight loading from checkpoint is handled by Lightning's
            # load_from_checkpoint(). The checkpoint strips 'model.' prefix and
            # excludes teacher/MRL keys. See claymodel/utils.py for the shared
            # weight-loading utility used by finetune factories.
        else:
            raise ValueError(
                f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            )

    def on_train_epoch_start(self) -> None:
        self.model.teacher.eval()

    @property
    def encoder(self) -> Encoder:
        """Access the encoder directly. Shortcut for self.model.encoder."""
        return self.model.encoder

    def forward(
        self, datacube: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(datacube)

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"],
            betas=(self.hparams["b1"], self.hparams["b2"]),
            fused=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5000,
            T_mult=1,
            eta_min=self.hparams["lr"] * 100,
            last_epoch=-1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, phase: str
    ) -> torch.Tensor:
        platform = batch["platform"][0]
        loss, reconstruction_loss, representation_loss = self(batch)

        losses: dict[str, torch.Tensor] = {
            "loss": loss,
            "rec_loss": reconstruction_loss,
            "rep_loss": representation_loss,
        }

        for loss_name, loss_value in losses.items():
            self.log(
                name=f"{phase}/{loss_name}",
                value=loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                name=f"{phase}_{platform}/{loss_name}",
                value=loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, phase="val")
