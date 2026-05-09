# Installation

## uv Installation (Recommended)

The easiest way to install Clay Foundation Model is via `uv`:

```bash
uv pip install git+https://github.com/Clay-foundation/model.git
```

This will install the `claymodel` package and all its dependencies. You can then import and use it in your Python code:

```python
from claymodel import load_metadata, normalize
from claymodel.api import load_model
import torch

# Load pretrained model (uses bundled metadata, inference-ready)
model = load_model(ckpt_path="path/to/clay-v1.5.ckpt")

# Prepare a datacube and generate embeddings
metadata = load_metadata()
sensor = "sentinel-2-l2a"
pixels = normalize(torch.randn(1, 10, 256, 256), sensor)
datacube = {
    "pixels": pixels,
    "time": torch.zeros(1, 4),
    "latlon": torch.zeros(1, 4),
    "gsd": torch.tensor(float(metadata[sensor].gsd)),
    "waves": torch.tensor(list(metadata[sensor].bands.wavelength.values())),
}

with torch.no_grad():
    encoded, *_ = model.encoder(datacube)
    embeddings = encoded[:, 0, :]  # [1, 1024]
```

If you want the `clay` CLI, install the `cli` extra:

```bash
uv pip install "claymodel[cli]"
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

Then install the dependencies with `uv`:

    uv pip install -e ".[dev]"

```{note}
The command above creates a local virtual environment and installs the project extras.
```

Finally, double-check that the libraries have been installed.

    uv run clay info
