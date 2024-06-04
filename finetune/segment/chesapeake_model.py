import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from box import Box
from torch import optim
from torchmetrics.classification import F1Score, MulticlassJaccardIndex

from finetune.segment.factory import Segmentor


class ChesapeakeSegmentor(L.LightningModule):
    def __init__(
        self,
        num_classes,
        feature_maps,
        ckpt_path,
        lr,
        wd,
        b1,
        b2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Segmentor(
            num_classes=num_classes, feature_maps=feature_maps, ckpt_path=ckpt_path
        )

        self.loss_fn = smp.losses.FocalLoss(mode="multiclass")
        self.iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="weighted",
        )
        self.f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
        )
        self.lr = lr

    @staticmethod
    def dice_loss(preds, target):
        return 1 - dice(preds, target, average="micro")

    def forward(self, datacube):
        platform = "naip"
        waves = torch.tensor([0.65, 0.56,  0.48, 0.842])
        gsd = torch.tensor(1.0)

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
            optimizer, T_0=1000, T_mult=1, eta_min=self.hparams.lr * 100, last_epoch=-1
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
        outputs = self(batch)
        outputs = F.interpolate(
            outputs, size=(224, 224), mode="bilinear", align_corners=False
        )

        loss = self.loss_fn(outputs, labels)
        iou = self.iou(outputs, labels)
        f1 = self.f1(outputs, labels)

        self.log(
            f"{phase}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            f"{phase}/iou",
            iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")
