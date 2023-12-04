import lightning as L
import torch

from src.geomae import GeoMAE
from vit_pytorch import MAE


class GeoMAEModule(L.LightningModule):
    def __init__(self, lr=1e-2):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model = GeoMAE()

    def forward(self, cube: dict[str, torch.Tensor]):
        return self.model(cube)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def shared_step(self, batch: dict[str, torch.Tensor], batch_idx: int, phase: str):
        cube = batch
        loss = self(cube)
        self.log(
            name=f"{phase}/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")
