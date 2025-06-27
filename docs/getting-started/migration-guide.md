# Migration Guide

This guide helps you migrate from the old development setup to the new pip-installable Clay package.

## What Changed?

Clay Foundation Model is now available as a proper Python package called `claymodel`. This means:

- ✅ **Easy installation**: `pip install git+https://github.com/Clay-foundation/model.git`
- ✅ **Clean imports**: `from claymodel.module import ClayMAEModule`
- ✅ **Better distribution**: No need to clone the entire repository for inference

## Import Changes

### Before (Old)
```python
# Old development imports
from src.datamodule import ClayDataModule
from src.module import ClayMAEModule
from src.model import ClayMAEEncoder
```

### After (New)
```python
# New package imports
from claymodel.datamodule import ClayDataModule
from claymodel.module import ClayMAEModule
from claymodel.model import ClayMAEEncoder
```

## Installation Methods

### For Inference Only (Recommended)

If you just want to use pretrained Clay models for generating embeddings:

```bash
# Install the package
pip install git+https://github.com/Clay-foundation/model.git

# Download weights
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
```

### For Development & Training

If you need to train models or contribute to development:

```bash
# Clone repository
git clone https://github.com/Clay-foundation/model.git
cd model

# Create environment
mamba env create --file environment.yml
mamba activate claymodel

# Install in development mode
pip install -e .
```

## Code Migration Examples

### Loading Pretrained Models

```python
# Before
import sys
sys.path.append('path/to/model/src')
from module import ClayMAEModule

# After
from claymodel.module import ClayMAEModule
```

### Generating Embeddings

```python
# Before
from src.module import ClayMAEModule

# After
from claymodel.module import ClayMAEModule

# Usage is the same
model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
embeddings = model.encoder(chips, timestamps, wavelengths)
```

### Training Workflows

Training workflows require the full development environment:

```python
# Before
from src.datamodule import ClayDataModule
from src.module import ClayMAEModule

# After (development install)
from claymodel.datamodule import ClayDataModule
from claymodel.module import ClayMAEModule
```

## Jupyter Notebooks

All tutorial notebooks have been updated to use the new imports:

- ✅ `docs/tutorials/embeddings.ipynb`
- ✅ `docs/tutorials/reconstruction.ipynb` 
- ✅ `docs/tutorials/wall-to-wall.ipynb`
- ✅ `docs/tutorials/inference.ipynb`

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'claymodel'`:

1. Ensure you've installed the package: `pip install git+https://github.com/Clay-foundation/model.git`
2. Restart your Python kernel/session
3. Check installation: `pip list | grep claymodel`

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
mamba env create --file environment.yml
mamba activate claymodel
pip install -e .
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