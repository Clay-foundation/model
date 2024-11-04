"""
Command line interface to run the neural network model!

From the project root directory, do:

    python classify.py fit --config configs/classify_eurosat.yaml

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

from lightning.pytorch.cli import LightningCLI

from finetune.classify.eurosat_datamodule import EuroSATDataModule  # noqa: F401
from finetune.classify.eurosat_model import EuroSATClassifier  # noqa: F401


# %%
def cli_main():
    """
    Command-line inteface to run Clasifier model with EuroSATDataModule.
    """
    cli = LightningCLI(
        EuroSATClassifier, EuroSATDataModule, save_config_kwargs={"overwrite": True}
    )
    return cli


# %%
if __name__ == "__main__":
    cli_main()

    print("Done!")
