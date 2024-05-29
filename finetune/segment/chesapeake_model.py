import yaml

import torch
import torch.nn.functional as F
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch import optim
from torchmetrics.classification import F1Score, MulticlassJaccardIndex


from finetune.segment.factory import Segmentor


class ChesapeakeSegmentor(L.LightningModule):
    def __init__(self, 
        metadata_path="configs/metadata.yaml", 
        num_classes=7, 
        feature_maps=[3, 5, 7, 11], 
        ckpt_path="checkpoints/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt",
        lr=1e-3,
    ):
        super().__init__()
        self.model = Segmentor(num_classes=num_classes, feature_maps=feature_maps, ckpt_path=ckpt_path)
        self.loss_fn = smp.losses.FocalLoss(mode="multiclass")
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=7)
        # self.loss_fn = smp.losses.DiceLoss(mode="multiclass")
        # self.loss_fn = smp.losses.LovaszLoss(mode="multiclass")
        # self.loss_fn = smp.losses.TverskyLoss(mode="multiclass", alpha=0.3, beta=0.7)
        self.metrics = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="weighted",
        )
        self.f1_score = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
        )
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.lr = lr

    @staticmethod
    def dice_loss(preds, target):
        return 1 - dice(preds, target, average="micro")

    def forward(self, datacube):
        platform = "naip"
        waves = torch.tensor(list(self.metadata[platform].bands.wavelength.values()))
        gsd = torch.tensor(self.metadata[platform].gsd)
        
        # Forward pass through the network
        return self.model({
            "pixels": datacube["pixels"],
            "time": datacube["time"],
            "latlon": datacube["latlon"],
            "gsd": gsd,
            "waves": waves,
        })

    def configure_optimizers(self):
        optimizer = optim.AdamW([param for name, param in self.model.named_parameters() if param.requires_grad], lr=self.lr)
        return optimizer

    
    def shared_step(self, batch, batch_idx, phase):
        labels = batch['label'].long()
        outputs = self(batch)
        outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)

        loss = self.loss_fn(outputs, labels)
        metrics = self.metrics(outputs, labels)
        f1_score = self.f1_score(outputs, labels)

        self.log(f'{phase}/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{phase}/metrics', metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{phase}/f1_score', f1_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")