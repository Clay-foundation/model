"""
Command line interface to run the neural network model!

From the project root directory, do:

    python segment.py fit --config configs/segment_chesapeake.yaml

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

from lightning.pytorch.cli import LightningCLI

from finetune.segment.chesapeake_datamodule import (
    ChesapeakeDataModule,
)
from finetune.segment.chesapeake_model import (
    ChesapeakeSegmentor,
)


def cli_main() -> LightningCLI:
    """
    Command-line inteface to run Segmentation Model with ChesapeakeDataModule.
    """
    return LightningCLI(
        ChesapeakeSegmentor,
        ChesapeakeDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()

    print("Done!")
