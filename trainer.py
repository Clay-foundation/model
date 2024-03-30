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

from src.callbacks_wandb import LogDINOPredictions
from src.datamodule import ClayDataModule, GeoTIFFDataPipeModule  # noqa: F401
from src.model_clay import CLAYDino, CLAYModule
from src.model_vit import ViTLitModule  # noqa: F401


class CLAYDinoWrapper(CLAYDino):
    def __init__(self, use_mean=False, **kwargs):
        self.use_mean = use_mean
        student = CLAYModule(**kwargs)
        teacher = CLAYModule(**kwargs)
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad = False
        super().__init__(
            student_clay=student,
            teacher_clay=teacher,
            output_dim=768,
            momentum_teacher=0.996,
            teacher_temp=0.04,
            student_temp=0.1,
            center_momentum=0.9,
            use_mean=self.use_mean,
            **kwargs,
        )


def cli_main(
    save_config_callback=None,
    seed_everything_default=42,
    trainer_defaults: dict = {
        "accelerator": "auto",
        "devices": "auto",  # "strategy": "ddp_find_unused_parameters_true",
        "precision": "bf16-mixed",
        "log_every_n_steps": 1,
        "max_epochs": 10000,
        "accumulate_grad_batches": 15,
        "default_root_dir": "s3://clay-model-ckpt/v0.2/",
        "callbacks": [
            ModelCheckpoint(
                dirpath="checkpoints/",
                auto_insert_metric_name=False,
                filename="Bali-DINO_epoch-{epoch:02d}_val-loss-{val/loss:.2f}",
                monitor="val/loss",
                mode="min",
                save_last=True,
                save_top_k=4,
                save_weights_only=False,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            LogDINOPredictions(),
        ],
        "logger": [WandbLogger(project="Bali-v0", log_model=False)],
        "plugins": [AsyncCheckpointIO()],
    },
    args: ArgsType = None,
):
    """
    Command-line interface to run CLAYDino with ClayDataModule.
    """
    cli = LightningCLI(
        model_class=CLAYDinoWrapper,
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
