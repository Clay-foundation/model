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
from claymodel import load_metadata, normalize
from claymodel.api import load_model

# Load model (uses bundled metadata, sets mask_ratio=0 for inference)
model = load_model(ckpt_path="clay-v1.5.ckpt")

# Load sensor metadata
metadata = load_metadata()

# Prepare Sentinel-2 data
sensor = "sentinel-2-l2a"
pixels = torch.randn(1, 10, 256, 256)  # [batch, bands, height, width]

# Normalize using sensor statistics (per-band z-score)
pixels = normalize(pixels, sensor, metadata=metadata)

# Build datacube with required keys
datacube = {
    "pixels": pixels,
    "time": torch.zeros(1, 4),     # (week_sin, week_cos, hour_sin, hour_cos)
    "latlon": torch.zeros(1, 4),   # (lat_sin, lat_cos, lon_sin, lon_cos)
    "gsd": torch.tensor(float(metadata[sensor].gsd)),
    "waves": torch.tensor(list(metadata[sensor].bands.wavelength.values())),
}

# Generate embeddings
with torch.no_grad():
    encoded, *_ = model.encoder(datacube)
    embeddings = encoded[:, 0, :]  # CLS token

print(f"Embeddings shape: {embeddings.shape}")  # [1, 1024]
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
