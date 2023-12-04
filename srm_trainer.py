"""
Command line interface to run the neural network model!

From the project root directory, do:

    python trainer.py fit

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.plugins.io import AsyncCheckpointIO
from lightning.pytorch.loggers import WandbLogger

from src.model_geomae import GeoMAEModule
from src.srm_datamodule import ClayDataModule


def main():
    model = GeoMAEModule(lr=1e-2)
    dm = ClayDataModule(batch_size=128, num_workers=8)
    dm.setup()

    wandb_logger = WandbLogger(project="CLAY-v0", log_model=False)
    ckpt_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
        dirpath="checkpoints/",
        auto_insert_metric_name=False,
        filename="mae_epoch-{epoch:02d}_val-loss-{val/loss:.3f}",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=20,
        logger=[wandb_logger],
        callbacks=[ckpt_callback],
        log_every_n_steps=1,
    )

    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )


# %%
def cli_main(
    save_config_callback=None,
    seed_everything_default=42,
    trainer_defaults: dict = {
        "accelerator": "auto",
        "callbacks": [
            ModelCheckpoint(
                dirpath="checkpoints/",
                auto_insert_metric_name=False,
                filename="mae_epoch-{epoch:02d}_val-loss-{val/loss:.2f}",
                monitor="val/loss",
                mode="min",
                save_last=False,
                save_top_k=1,
                save_weights_only=True,
            ),
        ],
        "logger": WandbLogger(project="CLAY-v0", log_model=False),
        "plugins": [AsyncCheckpointIO()],
        "precision": "bf16-mixed",
        "max_epochs": 20,
    },
    args: ArgsType = None,
):
    """
    Command-line inteface to run ViTLitModule with BaseDataModule.
    """
    cli = LightningCLI(
        model_class=GeoMAEModule,
        datamodule_class=ClayDataModule,
        save_config_callback=save_config_callback,
        seed_everything_default=seed_everything_default,
        trainer_defaults=trainer_defaults,
        args=args,
    )
    return cli


if __name__ == "__main__":
    main()
    print("Done!")
