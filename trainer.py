"""Lightning CLI entry point for training."""

from lightning.pytorch.cli import LightningCLI

from claymodel.model import configure_training_defaults
from claymodel.module import ClayMAEModule
from training.datamodule import ClayDataModule

configure_training_defaults()


def cli_main() -> LightningCLI:
    """Run ClayMAE with ClayDataModule."""
    return LightningCLI(
        ClayMAEModule,
        ClayDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
