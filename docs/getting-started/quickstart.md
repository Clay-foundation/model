# Quick Start Guide

Get started with Clay Foundation Model in 5 minutes!

## 1. Install

```bash
uv pip install git+https://github.com/Clay-foundation/model.git
```

## 2. Download Weights

```bash
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
```

## 3. Generate Embeddings

```python
import torch
from claymodel import embed, load_model

# Load encoder (no teacher download, fast startup)
encoder = load_model("large", ckpt_path="clay-v1.5.ckpt")

# Generate embeddings from Sentinel-2 data
pixels = torch.randn(1, 10, 256, 256)  # [batch, bands, height, width]
result = embed(pixels, sensor="sentinel-2-l2a", model=encoder)

print(f"Embeddings shape: {result.shape}")  # [1, 1024]
```

## 4. Next Steps

- **Explore tutorials**: [Embeddings](../tutorials/embeddings.ipynb) | [Reconstruction](../tutorials/reconstruction.ipynb) | [Wall-to-Wall](../tutorials/wall-to-wall.ipynb)
- **Finetune for your task**: [Classification](../finetune/classify.md) | [Segmentation](../finetune/segment.md) | [Regression](../finetune/regression.md)
- **Learn the model**: [Architecture & Training](../release-notes/specification.md)

## Supported Sensors

Clay v1.5 is **sensor-agnostic** and works with **any satellite instrument**! Currently supported sensors include:

| Sensor | Bands | Resolution | Description |
|--------|-------|------------|-------------|
| Sentinel-2 L2A | 10 | 10m | Optical multispectral |
| Landsat C2 L1/L2 | 6 | 30m | Optical multispectral |
| NAIP | 4 | 1m | Aerial RGB + NIR |
| LINZ | 3 | 0.5m | Aerial RGB |
| Sentinel-1 | 2 | 10m | SAR (VV, VH) |
| MODIS | 7 | 500m | Global surface reflectance |

```yaml
your-satellite:
  band_order: [blue, green, red, nir]    # Your band names
  gsd: 10.0                              # Resolution in meters
  bands:
    wavelength: {blue: 0.485, green: 0.560, red: 0.660, nir: 0.835}  # μm
    mean: {blue: 1200, green: 1400, red: 1600, nir: 2800}           # Normalization
    std: {blue: 400, green: 450, red: 500, nir: 650}                # Statistics
```

## Need Help?

- 📖 **Full Documentation**: [clay-foundation.github.io/model](https://clay-foundation.github.io/model)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Clay-foundation/model/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Clay-foundation/model/discussions)
