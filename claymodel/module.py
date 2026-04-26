__all__ = ["ClayMAEModule"]

import random
from collections.abc import Mapping
from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F

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
        else:
            raise ValueError(
                f"Invalid model size {model_size}. "
                f"Expected one of {list(model_map.keys())}"
            )

    def on_train_epoch_start(self) -> None:
        self.model.teacher.eval()

    @property
    def encoder(self) -> Encoder:
        """Access the encoder directly. Shortcut for self.model.encoder."""
        return self.model.encoder

    def forward(
        self, datacube: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _resolve_platform(
        self, batch: Mapping[str, object]
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """Extract platform name, wavelengths, and GSD from a batch."""
        platforms = batch["platform"]
        assert isinstance(platforms, list)
        platform = platforms[0]
        assert isinstance(platform, str)
        meta = self.metadata[platform]
        waves = torch.tensor(list(meta.bands.wavelength.values()))
        gsd = torch.tensor(meta.gsd)
        return platform, waves, gsd

    def _channel_dropout(
        self,
        pixels: torch.Tensor,
        latlon: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly drop channels as training augmentation.

        10% chance to zero all channels, 20% chance to zero half.
        Only applied to samples with nonzero latlon.
        """
        pixels = pixels.clone()
        batch_size, channels, _, _ = pixels.size()

        for i in range(batch_size):
            if torch.any(latlon[i] != 0):
                rand_val = random.random()
                if rand_val < 0.10:
                    pixels[i, :, :, :] = 0
                elif rand_val < 0.30:
                    drop_idx = torch.randperm(channels)[: channels // 2]
                    pixels[i, drop_idx, :, :] = 0

        return pixels

    def _teacher_target(
        self,
        pixels: torch.Tensor,
        platform: str,
    ) -> torch.Tensor:
        """Compute teacher representation target from RGB bands."""
        with torch.no_grad():
            if platform == "sentinel-1-rtc":
                r = pixels[:, 0, :, :]
                g = pixels[:, 1, :, :]
                b = (r + g) / 2
                rgb = torch.stack((r, g, b), dim=1)
            else:
                indices = self.metadata[platform].rgb_indices
                rgb = pixels[:, indices, :, :]
            rgb = self.model.teacher_resize(rgb)
            return self.model.teacher(rgb)

    def _representation_loss(
        self,
        cls_token: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute representation loss (proj or MRL)."""
        if self.model.matryoshka:
            representations = self.model.mrl(cls_token)
            return self.model.mrl_loss(representations, target)
        representations = self.model.proj(cls_token)
        return 1.0 - F.cosine_similarity(representations, target).mean()

    def _log_losses(
        self,
        phase: str,
        platform: str,
        loss: torch.Tensor,
        rec_loss: torch.Tensor,
        rep_loss: torch.Tensor,
    ) -> None:
        losses = {"loss": loss, "rec_loss": rec_loss, "rep_loss": rep_loss}
        for name, value in losses.items():
            self.log(
                name=f"{phase}/{name}",
                value=value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                name=f"{phase}_{platform}/{name}",
                value=value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        platform, waves, gsd = self._resolve_platform(batch)
        pixels = self._channel_dropout(batch["pixels"], batch["latlon"])

        datacube = {
            "pixels": pixels,
            "time": batch["time"],
            "latlon": batch["latlon"],
            "gsd": gsd,
            "waves": waves,
        }

        encoded, decoded_pixels, masked_matrix, *_ = self.model(datacube)

        rec_loss = self.model.per_pixel_loss(
            batch["pixels"], decoded_pixels, masked_matrix
        )
        if platform == "modis":
            rec_loss = rec_loss / 10

        target = self._teacher_target(batch["pixels"], platform)
        rep_loss = self._representation_loss(encoded[:, 0, :], target)

        loss = 0.9 * rec_loss + 0.1 * rep_loss
        self._log_losses("train", platform, loss, rec_loss, rep_loss)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        platform, waves, gsd = self._resolve_platform(batch)

        datacube = {
            "pixels": batch["pixels"],
            "time": batch["time"],
            "latlon": batch["latlon"],
            "gsd": gsd,
            "waves": waves,
        }

        encoded, decoded_pixels, masked_matrix, *_ = self.model(datacube)

        rec_loss = self.model.per_pixel_loss(
            batch["pixels"], decoded_pixels, masked_matrix
        )
        if platform == "modis":
            rec_loss = rec_loss / 10

        target = self._teacher_target(batch["pixels"], platform)
        rep_loss = self._representation_loss(encoded[:, 0, :], target)

        loss = 0.9 * rec_loss + 0.1 * rep_loss
        self._log_losses("val", platform, loss, rec_loss, rep_loss)
        return loss
