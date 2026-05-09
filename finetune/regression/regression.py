"""
Command line interface to run the neural network model!

From the project root directory, do:

    python regression.py fit --config configs/regression_biomasters.yaml

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

from lightning.pytorch.cli import LightningCLI

from finetune.regression.biomasters_datamodule import (
    BioMastersDataModule,
)
from finetune.regression.biomasters_model import (
    BioMastersClassifier,
)


def cli_main() -> LightningCLI:
    """
    Command-line inteface to run Regression with BioMastersDataModule.
    """
    return LightningCLI(
        BioMastersClassifier,
        BioMastersDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()

    print("Done!")
