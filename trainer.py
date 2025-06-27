"""
Command line interface to run the neural network model!

From the project root directory, do:

    python trainer.py fit

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

from lightning.pytorch.cli import LightningCLI

from claymodel.datamodule import ClayDataModule  # noqa: F401
from claymodel.module import ClayMAEModule  # noqa: F401


# %%
def cli_main():
    """
    Command-line inteface to run ClayMAE with ClayDataModule.
    """
    cli = LightningCLI(
        ClayMAEModule, ClayDataModule, save_config_kwargs={"overwrite": True}
    )
    return cli


# %%
if __name__ == "__main__":
    cli_main()

    print("Done!")
