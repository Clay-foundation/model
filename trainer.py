"""
Command line interface to run the neural network model!

From the project root directory, do:

    python trainer.py fit

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""
from lightning.pytorch.callbacks import (
    LearningRateMonitor,  # noqa: F401
    ModelCheckpoint,
)
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.plugins.io import AsyncCheckpointIO

from src.callbacks import LogIntermediatePredictions  # noqa: F401
from src.datamodule import GeoTIFFDataPipeModule  # noqa: F401
from src.model import CLAYModule
from src.model_vit import ViTLitModule  # noqa: F401
from src.srm_datamodule import ClayDataModule


# %%
def cli_main(
    save_config_callback=None,
    seed_everything_default=42,
    trainer_defaults: dict = {
        "accelerator": "auto",
        "callbacks": [
            ModelCheckpoint(
                # dirpath="checkpoints/",
                auto_insert_metric_name=False,
                filename="mae_epoch-{epoch:02d}_val-loss-{val/loss:.2f}",
                monitor="val/loss",
                mode="min",
                save_last=True,
                save_top_k=3,
                save_weights_only=True,
                verbose=True,
            ),
            # LearningRateMonitor(logging_interval="step"),
            # LogIntermediatePredictions(logger=wandb_logger),
        ],
        "logger": False,  # WandbLogger(project="CLAY-v0", log_model=False)
        "plugins": [AsyncCheckpointIO()],
        "precision": "bf16-mixed",
        "max_epochs": 20,
        "log_every_n_steps": 1,
        "accumulate_grad_batches": 20,
    },
    args: ArgsType = None,
):
    """
    Command-line inteface to run CLAYModule with ClayDataModule.
    """
    cli = LightningCLI(
        model_class=CLAYModule,
        datamodule_class=ClayDataModule,
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
