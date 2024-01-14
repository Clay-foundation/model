# Installation

## Cloud Environments

Launch into a [JupyterLab](https://jupyterlab.readthedocs.io) environment on

| [Binder](https://mybinder.readthedocs.io/en/latest) | [Planetary Computer](https://planetarycomputer.microsoft.com) | [SageMaker Studio Lab](https://studiolab.sagemaker.aws) |
|:--:|:--:|:--:|
| [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Clay-foundation/model/main) | [![Open on Planetary Computer](https://img.shields.io/badge/Open-Planetary%20Computer-black?style=flat&logo=microsoft)](https://pccompute.westeurope.cloudapp.azure.com/compute/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FClay-foundation%2Fmodel&urlpath=lab%2Ftree%2Fmodel%2Fplaceholder.ipynb&branch=main) | [![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/Clay-foundation/model/blob/main/placeholder.ipynb) |


## Local Environments

Start by cloning this [repo-url](https://github.com/Clay-foundation/model)

    git clone https://github.com/Clay-foundation/model

Then we recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation.html)
to install the dependencies.

A virtual environment will also be created with Python and
[JupyterLab](https://github.com/jupyterlab/jupyterlab) installed.

    cd model
    mamba env create --file environment.yml

Activate the virtual environment first.

    mamba activate claymodel

## Advanced

See [Readme](../../README.md) on model root for more details.
