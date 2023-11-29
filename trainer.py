"""
Command line interface to run the neural network model!

From the project root directory, do:

    python trainer.py fit

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.plugins.io import AsyncCheckpointIO

from src.datamodule import GeoTIFFDataPipeModule
from src.model_vit import ViTLitModule


# %%
def cli_main(
    save_config_callback=None,
    seed_everything_default=42,
    trainer_defaults: dict = {
        "callbacks": [
            ModelCheckpoint(
                # dirpath="checkpoints/",
                auto_insert_metric_name=False,
                filename="vit_epoch-{epoch:02d}_train_loss-{train/loss:.2f}",
                monitor="train/loss",
                mode="min",
                save_last=True,
                save_top_k=3,
                save_weights_only=True,
            ),
        ],
        "logger": False,
        "plugins": [AsyncCheckpointIO()],
        "precision": "bf16-mixed",
    },
    args: ArgsType = None,
):
    """
    Command-line inteface to run ViTLitModule with BaseDataModule.
    """
    cli = LightningCLI(
        model_class=ViTLitModule,
        datamodule_class=GeoTIFFDataPipeModule,
        save_config_callback=save_config_callback,
        seed_everything_default=seed_everything_default,
        trainer_defaults=trainer_defaults,
        args=args,
    )
    return cli


# %%
if __name__ == "__main__":
    cli_main()
    print("Done!")
