from typing import Literal

import lightning as L
import torch
import yaml
from box import Box

from src.model import clay_mae_base, clay_mae_large, clay_mae_small, clay_mae_tiny


class ClayMAEModule(L.LightningModule):
    def __init__(  # noqa: PLR0913
        self,
        model_size="base",
        mask_ratio=0.75,
        norm_pix_loss=False,
        patch_size=8,
        shuffle=False,
        metadata_path="configs/metadata.yaml",
        teacher="samvit_base_patch16.sa1b",
        dolls=[16, 32, 64, 128, 256, 768],
        doll_weights=[1, 1, 1, 1, 1, 1],
        lr=1e-5,
        wd=0.05,
        b1=0.9,
        b2=0.95,
        embeddings_level: Literal["mean", "patch", "group"] = "mean",
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
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
            }
            self.model = model_map[model_size](**model_args)
        else:
            raise ValueError(
                f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            )

    def on_train_epoch_start(self):
        self.model.teacher.eval()

    def forward(self, datacube: dict[str, torch.Tensor]):
        return self.model(datacube)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
            fused=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=1, eta_min=self.hparams.lr * 100, last_epoch=-1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch: dict[str, torch.Tensor], batch_idx: int, phase: str):
        platform = batch["platform"][0]
        loss, reconstruction_loss, representation_loss = self(batch)

        losses = {
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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")
