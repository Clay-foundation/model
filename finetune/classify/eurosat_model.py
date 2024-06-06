import re
import yaml
from pathlib import Path

import torch
import lightning as L
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
from einops import rearrange
from PIL import Image
from box import Box
from torchvision.transforms import v2
from torchgeo.datasets import EuroSAT
from torch import nn

from torch import optim
from torchmetrics import Accuracy
from finetune.classify.factory import Classifier

class EuroSATClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes,
        ckpt_path,
        lr,
        wd,
        b1,
        b2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Classifier(num_classes=num_classes, ckpt_path=ckpt_path)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        

    def forward(self, datacube):
        platform = "sentinel-2-l2a"
        waves = torch.tensor([0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19])
        gsd = torch.tensor(10.)

        # Forward pass through the network
        return self.model(
            {
                "pixels": datacube["pixels"],
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            }
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=1, eta_min=self.hparams.lr * 100, last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch, batch_idx, phase):
        labels = batch["label"].long()
        logits = self(batch)
        loss = self.loss_fn(logits, labels)
        score = self.accuracy(logits, labels)
        
        self.log(
            f"{phase}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/score",
            score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")