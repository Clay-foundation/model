# Installation

## uv Installation (Recommended)

The easiest way to install Clay Foundation Model is via `uv`:

```bash
uv pip install git+https://github.com/Clay-foundation/model.git
```

This will install the `claymodel` package and all its dependencies. You can then import and use it in your Python code:

```python
import torch
from claymodel import embed, load_model

# Load encoder (no teacher download, fast startup)
encoder = load_model("large", ckpt_path="path/to/clay-v1.5.ckpt")

# Generate embeddings from Sentinel-2 data
pixels = torch.randn(1, 10, 256, 256)  # [batch, bands, height, width]
result = embed(pixels, sensor="sentinel-2-l2a", model=encoder)
print(f"Embeddings shape: {result.shape}")  # [1, 1024]
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
