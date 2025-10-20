# Clay Foundation Model

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://clay-foundation.github.io/model)
[![Deploy Book Status](https://github.com/Clay-foundation/model/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/Clay-foundation/model/actions/workflows/deploy-docs.yml)

An open source AI model and interface for Earth.

## License

Clay Model is licensed under the [Apache](LICENSE). This applies to the source code as well as the trained model weights.

The Documentation is licensed under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Quickstart

Launch into a [JupyterLab](https://jupyterlab.readthedocs.io) environment on

| [Binder](https://mybinder.readthedocs.io/en/latest) | [SageMaker Studio Lab](https://studiolab.sagemaker.aws) |
|:--:|:--:|
| [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Clay-foundation/model/main) | [![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/Clay-foundation/model/blob/main/docs/tutorials/wall-to-wall.ipynb) |

## Installation

### Pip Installation (Recommended)

The easiest way to install Clay Foundation Model is via pip:

    pip install git+https://github.com/Clay-foundation/model.git

This will install the `claymodel` package and all its dependencies. You can then import and use it in your Python code:

```python
from claymodel.datamodule import ClayDataModule
from claymodel.module import ClayMAEModule
```

### Development Installation

For development or advanced usage, you can set up the full development environment:

To help out with development, start by cloning this [repo-url](/../../)

    git clone https://github.com/Clay-foundation/model.git
    cd model


Then we recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
to install the dependencies. A virtual environment will also be created with Python and
[JupyterLab](https://github.com/jupyterlab/jupyterlab) installed.

    mamba env create --file environment.yml

> [!NOTE]
> The command above has been tested on Linux devices with CUDA GPUs.

Activate the virtual environment first.

    mamba activate claymodel

Finally, double-check that the libraries have been installed.

    mamba list


## Usage

### Running jupyter lab

    mamba activate claymodel
    python -m ipykernel install --user --name claymodel  # to install virtual env properly
    jupyter kernelspec list --json                       # see if kernel is installed
    jupyter lab &


### Running the model

The neural network model can be run via
[LightningCLI v2](https://pytorch-lightning.medium.com/introducing-lightningcli-v2supercharge-your-training-c070d43c7dd6).

> [!NOTE]
> If you installed via pip, you'll need to clone the repository to access the trainer script and config files.

To check out the different options available, and look at the hyperparameter
configurations, run:

    python trainer.py --help

To quickly test the model on one batch in the validation set:

    python trainer.py fit --model ClayMAEModule --data ClayDataModule --config configs/config.yaml --trainer.fast_dev_run=True

To train the model:

    python trainer.py fit --model ClayMAEModule --data ClayDataModule --config configs/config.yaml

More options can be found using `python trainer.py fit --help`, or at the
[LightningCLI docs](https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html).

## Contributing

### Writing documentation

Our Documentation uses [Jupyter Book](https://jupyterbook.org/intro.html).

Install it with:
```bash
pip install -U jupyter-book
```

Then build it with:
```bash
jupyter-book build docs/
```

You can preview the site locally with:
```bash
python -m http.server --directory _build/html
```

There is a GitHub Action on `.github/workflows/deploy-docs.yml` that builds the site and pushes it to GitHub Pages.
