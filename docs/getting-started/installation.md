# Installation

## Cloud Environments

Launch into a [JupyterLab](https://jupyterlab.readthedocs.io) environment on

| [Binder](https://mybinder.readthedocs.io/en/latest) | [SageMaker Studio Lab](https://studiolab.sagemaker.aws) |
|:--:|:--:|
| [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Clay-foundation/model/main) | [![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/Clay-foundation/model/blob/main/docs/tutorials/clay-v1-wall-to-wall.ipynb) |

## Local Environments

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
