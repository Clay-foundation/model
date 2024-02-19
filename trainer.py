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
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.io import AsyncCheckpointIO

from src.callbacks_wandb import (  # noqa: F401
    LogIntermediatePredictions,
    LogMAEReconstruction,
)
from src.datamodule import ClayDataModule, GeoTIFFDataPipeModule  # noqa: F401
from src.model_clay import CLAYModule
from src.model_vit import ViTLitModule  # noqa: F401


# %%
def cli_main(
    save_config_callback=None,
    seed_everything_default=42,
    trainer_defaults: dict = {
        "accelerator": "auto",
        "devices": "auto",
        "strategy": "ddp",
        "precision": "bf16-mixed",
        "log_every_n_steps": 1,
        "max_epochs": 100,
        "accumulate_grad_batches": 5,
        "callbacks": [
            ModelCheckpoint(
                dirpath="checkpoints/",
                auto_insert_metric_name=False,
                filename="mae_epoch-{epoch:02d}_val-loss-{val/loss:.2f}",
                monitor="val/loss",
                mode="min",
                save_last=True,
                save_top_k=2,
                save_weights_only=True,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            LogIntermediatePredictions(),
        ],
        "logger": [WandbLogger(project="CLAY-v0", log_model=False)],
        "plugins": [AsyncCheckpointIO()],
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
    # tracemalloc.start()
    cli_main()

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #    print(stat)
    print("Done!")
