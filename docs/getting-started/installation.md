# Installation

## Pip Installation (Recommended)

The easiest way to install Clay Foundation Model is via pip:

```bash
pip install git+https://github.com/Clay-foundation/model.git
```

This will install the `claymodel` package and all its dependencies. You can then import and use it in your Python code:

```python
from claymodel.datamodule import ClayDataModule
from claymodel.module import ClayMAEModule

# Load pretrained model
model = ClayMAEModule.load_from_checkpoint("path/to/clay-v1.5.ckpt")

# Generate embeddings
embeddings = model.encoder(chips)
```

### Using Pretrained Weights

Download the Clay v1.5 weights from [Hugging Face](https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt):

```bash
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
```

## Cloud Environments

Launch into a [JupyterLab](https://jupyterlab.readthedocs.io) environment on

| [Binder](https://mybinder.readthedocs.io/en/latest) | [SageMaker Studio Lab](https://studiolab.sagemaker.aws) |
|:--:|:--:|
| [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Clay-foundation/model/main) | [![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/Clay-foundation/model/blob/main/docs/tutorials/wall-to-wall.ipynb) |

## Development Installation

For development, training, or advanced usage, you can set up the full development environment:

Start by cloning this [repo-url](https://github.com/Clay-foundation/model)

    git clone https://github.com/Clay-foundation/model
    cd model

Then we recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
to install the dependencies. A virtual environment will also be created with Python and
[JupyterLab](https://github.com/jupyterlab/jupyterlab) installed.

    mamba env create --file environment.yml

```{note}
The command above has been tested on Linux devices with CUDA GPUs.
```

Activate the virtual environment first.

    mamba activate claymodel

Finally, double-check that the libraries have been installed.

    mamba list
