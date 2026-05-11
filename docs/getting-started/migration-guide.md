# Migration Guide

This guide helps you migrate from the old development setup to the new pip-installable Clay package.

## What Changed?

Clay Foundation Model is now available as a proper Python package called `claymodel`. This means:

- ✅ **Easy installation**: `uv pip install git+https://github.com/Clay-foundation/model.git`
- ✅ **Clean imports**: `from claymodel import load_model, embed`
- ✅ **Better distribution**: No need to clone the entire repository for inference

## Import Changes

### Before (Old)
```python
# Old development imports
from src.datamodule import ClayDataModule
from src.module import ClayMAEModule
from src.model import Encoder
```

### After (New)
```python
# Inference imports
from claymodel import load_model, embed, Encoder

# Training imports (requires full repo + dev install):
from claymodel.module import ClayMAEModule
from training.datamodule import ClayDataModule
```

## Installation Methods

### For Inference Only (Recommended)

If you just want to use pretrained Clay models for generating embeddings:

```bash
# Install the package
uv pip install git+https://github.com/Clay-foundation/model.git

# Download weights
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
```

### For Development & Training

If you need to train models or contribute to development:

```bash
# Clone repository
git clone https://github.com/Clay-foundation/model.git
cd model

# Install in development mode
uv pip install -e ".[dev]"
```

## Code Migration Examples

### Generating Embeddings

```python
# Before
from src.module import ClayMAEModule
model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
model.eval()
model.model.encoder.mask_ratio = 0.0

# After — no teacher download, fast startup
from claymodel import embed, load_model
encoder = load_model("large", ckpt_path="clay-v1.5.ckpt")
result = embed(pixels, sensor="sentinel-2-l2a", model=encoder)
```

### Training Workflows

Training workflows require the full development environment:

```python
# Before
from src.datamodule import ClayDataModule
from src.module import ClayMAEModule

# After (development install, requires `uv pip install "claymodel[train]"`)
from training.datamodule import ClayDataModule
from claymodel.module import ClayMAEModule
```

## Jupyter Notebooks

Tutorial notebooks are being updated to use the new package imports. Status:

- ✅ `docs/tutorials/wall-to-wall.ipynb` — updated, uses public STAC data
- ⚠️ `docs/tutorials/embeddings.ipynb` — imports fixed, requires local training data
- ⚠️ `docs/tutorials/reconstruction.ipynb` — imports fixed, requires local training data
- ⚠️ `docs/tutorials/inference.ipynb` — import fixed, uses Clay v1 (update to v1.5 planned)

For the simplest embedding workflow, use the new API: `from claymodel import embed`

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'claymodel'`:

1. Ensure you've installed the package: `uv pip install git+https://github.com/Clay-foundation/model.git`
2. Restart your Python kernel/session
3. Check installation: `uv pip show claymodel`

### Old Notebooks

If you have old notebooks with `from src` imports:

1. Replace `from src.` with `from claymodel.`
2. Ensure claymodel is installed
3. Update any hardcoded paths

### Development Setup

For development work, you still need the full repository:

```bash
git clone https://github.com/Clay-foundation/model.git
cd model
uv pip install -e ".[dev]"
```

## Benefits of Migration

- **Easier deployment**: No need to manage source paths
- **Cleaner environments**: Proper dependency management
- **Better portability**: Code works across different setups
- **Professional packaging**: Follows Python packaging best practices

## Need Help?

- 📖 **Documentation**: [clay-foundation.github.io/model](https://clay-foundation.github.io/model)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Clay-foundation/model/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Clay-foundation/model/discussions)
