import lightning as L
import torch
from torch import nn, optim
from torchmetrics import Accuracy

from finetune.classify.factory import Classifier


class EuroSATClassifier(L.LightningModule):
    """
    LightningModule for training and evaluating a classifier on the EuroSAT
    dataset.

    Args:
        num_classes (int): Number of classes for classification.
        ckpt_path (str): Clay MAE pretrained checkpoint path.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for the optimizer.
        b1 (float): Beta1 parameter for the Adam optimizer.
        b2 (float): Beta2 parameter for the Adam optimizer.
    """

    def __init__(self, num_classes, ckpt_path, lr, wd, b1, b2):  # noqa: PLR0913
        super().__init__()
        self.save_hyperparameters()
        self.model = Classifier(num_classes=num_classes, ckpt_path=ckpt_path)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, datacube):
        """
        Forward pass through the classifier.

        Args:
            datacube (dict): A dictionary containing the input datacube
            and meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The output logits from the classifier.
        """
        # Wavelengths for Sentinel 2 bands of EuroSAT dataset
        waves = torch.tensor(
            [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]
        )
        gsd = torch.tensor(10.0)

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
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and learning rate
            scheduler.
        """
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
        """
        Perform a shared step for both training and validation.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.
            phase (str): The phase ('train' or 'val').

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
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
        """
        Perform a training step.

        Args:
            batch (dict): A batch of training data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        Args:
            batch (dict): A batch of validation data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        return self.shared_step(batch, batch_idx, "val")
