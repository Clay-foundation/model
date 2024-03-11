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
        "log_every_n_steps": 50,
        "max_epochs": 100,
        "accumulate_grad_batches": 5,
        "default_root_dir": "s3://clay-model-ckpt/v0.2/",
        "callbacks": [
            ModelCheckpoint(
                dirpath=None,
                auto_insert_metric_name=False,
                filename="mae_epoch-{epoch:02d}_val-loss-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_last=True,
                save_top_k=2,
                save_weights_only=False,    
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
        model_class=CLAYModule(
            model_size="small",
            mask_ratio=0.75,
            image_size=256,
            patch_size=16,
            lr=1e-5,
        ),
        datamodule_class=ClayDataModule(
            data_dir="data",
            batch_size=10,
            num_workers=8,
        ),
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
