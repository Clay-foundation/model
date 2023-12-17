# Clay Foundation Model

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://clay-foundation.github.io/model)
[![Deploy Book Status](https://github.com/Clay-foundation/model/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/Clay-foundation/model/actions/workflows/deploy-docs.yml)
[![Continuous Integration Tests Status](https://github.com/Clay-foundation/model/actions/workflows/test.yml/badge.svg)](https://github.com/Clay-foundation/model/actions/workflows/test.yml)

An open source AI model and interface for Earth

# Getting started

## Quickstart

Launch into a [JupyterLab](https://jupyterlab.readthedocs.io) environment on

| [Binder](https://mybinder.readthedocs.io/en/latest) | [Planetary Computer](https://planetarycomputer.microsoft.com) | [SageMaker Studio Lab](https://studiolab.sagemaker.aws) |
|:--:|:--:|:--:|
| [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Clay-foundation/model/main) | [![Open on Planetary Computer](https://img.shields.io/badge/Open-Planetary%20Computer-black?style=flat&logo=microsoft)](https://pccompute.westeurope.cloudapp.azure.com/compute/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FClay-foundation%2Fmodel&urlpath=lab%2Ftree%2Fmodel%2Fplaceholder.ipynb&branch=main) | [![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/Clay-foundation/model/blob/main/placeholder.ipynb) |

## Installation

### Basic

To help out with development, start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then we recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation.html)
to install the dependencies.
A virtual environment will also be created with Python and
[JupyterLab](https://github.com/jupyterlab/jupyterlab) installed.

    cd model
    mamba env create --file environment.yml

Activate the virtual environment first.

    mamba activate claymodel

Finally, double-check that the libraries have been installed.

    mamba list

### Advanced

This is for those who want full reproducibility of the virtual environment.
Create a virtual environment with just Python and conda-lock installed first.

    mamba create --name claymodel python=3.11 conda-lock=2.5.1
    mamba activate claymodel

Generate a unified [`conda-lock.yml`](https://github.com/conda/conda-lock) file
based on the dependency specification in `environment.yml`. Use only when
creating a new `conda-lock.yml` file or refreshing an existing one.

    conda-lock lock --mamba --file environment.yml --platform linux-64 --with-cuda=12.0

Installing/Updating a virtual environment from a lockile. Use this to sync your
dependencies to the exact versions in the `conda-lock.yml` file.

    conda-lock install --mamba --name claymodel conda-lock.yml

See also https://conda.github.io/conda-lock/output/#unified-lockfile for more
usage details.

## Usage

### Running jupyter lab

    mamba activate claymodel
    python -m ipykernel install --user --name claymodel  # to install virtual env properly
    jupyter kernelspec list --json                       # see if kernel is installed
    jupyter lab &


### Running the model

The neural network model can be ran via
[LightningCLI v2](https://pytorch-lightning.medium.com/introducing-lightningcli-v2supercharge-your-training-c070d43c7dd6).
To check out the different options available, and look at the hyperparameter
configurations, run:

    python trainer.py --help
    python trainer.py test --print_config

To quickly test the model on one batch in the validation set:

    python trainer.py validate --trainer.fast_dev_run=True

To train the model for a hundred epochs:

    python trainer.py fit --trainer.max_epochs=100

To generate embeddings from the pretrained model's encoder on 1024 images
(stored as a GeoParquet file with spatiotemporal metadata):

    python trainer.py predict --ckpt_path=checkpoints/last.ckpt \
                              --data.batch_size=1024 \
                              --data.data_path=s3://clay-tiles-02 \
                              --trainer.limit_predict_batches=1

More options can be found using `python trainer.py fit --help`, or at the
[LightningCLI docs](https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html).

### Running the datacube pipeline

How to run the data pipeline on AWS Batch Spot instances using
a [fetch-and-run](https://aws.amazon.com/blogs/compute/creating-a-simple-fetch-and-run-aws-batch-job/)
approach.

#### Prepare docker image in ECR

Build the docker image and push it to a ecr repository.

```bash
ecr_repo_id=12345
cd batch
docker build -t $ecr_repo_iud.dkr.ecr.us-east-1.amazonaws.com/fetch-and-run .

aws ecr get-login-password --profile clay --region us-east-1 | docker login --username AWS --password-stdin $ecr_repo_iud.dkr.ecr.us-east-1.amazonaws.com

docker push $ecr_repo_iud.dkr.ecr.us-east-1.amazonaws.com/fetch-and-run:latest
```
#### Prepare AWS batch

To prepare batch, we need to create a compute environment, job queue, and job
definition.

Example configurations for the compute environment and the job definition are
provided in the `batch` directory.

The `submit.py` script contains a loop for submitting jobs to the queue. An
alternative to this individual job submissions would be to use array jobs, but
for now the individual submissions are simpler and failures are easier to track.

#### Create ZIP file with the package to execute

Package the model and the inference script into a zip file. The `datacube.py`
script is the one that will be executed on the instances.

Put the scripts in a zip file and upload the zip package into S3 so that
the batch fetch and run can use it.

```bash
zip -FSrj "batch-fetch-and-run.zip" ./scripts/* -x "scripts/*.pyc"

aws s3api put-object --bucket clay-fetch-and-run-packages --key "batch-fetch-and-run.zip" --body "batch-fetch-and-run.zip"
```

#### Submit job

We can now submit a batch job to run the pipeline. The `submit.py` file
provides an example on how to sumbit jobs in python.
