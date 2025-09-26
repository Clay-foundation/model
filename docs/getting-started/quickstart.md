# Quick Start Guide

Get started with Clay Foundation Model in 5 minutes!

## 1. Install

```bash
pip install git+https://github.com/Clay-foundation/model.git
```

## 2. Download Weights

```bash
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
```

## 3. Generate Embeddings

```python
import yaml
import torch
from claymodel.module import ClayMAEModule

# Load model
model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
model.eval()

# Load sensor metadata
with open("configs/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)

# Prepare Sentinel-2 data
sensor = "sentinel-2-l2a"
chips = torch.randn(1, 10, 256, 256)  # [batch, bands, height, width]

# Get wavelengths from metadata (convert μm to nm)
wavelengths = []
for band in metadata[sensor]["band_order"]:
    wavelengths.append(metadata[sensor]["bands"]["wavelength"][band] * 1000)
wavelengths = torch.tensor([wavelengths], dtype=torch.float32)

timestamps = torch.zeros(1, 4)  # [week, hour, lat, lon] - can be zeros

# Generate embeddings
with torch.no_grad():
    embeddings = model.encoder(chips, timestamps, wavelengths)

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

## 4. Handle Missing Data

Clay automatically handles nodata pixels (clouds, shadows, projection gaps):

```python
from claymodel.utils import create_datacube_with_mask

# Automatic nodata detection
datacube = create_datacube_with_mask(
    pixels=chips,              # Your data with nodata values  
    time=timestamps,
    latlon=torch.zeros(1, 4),
    platform="sentinel-2-l2a"
)

# Or provide custom masks
cloud_mask = load_your_cloud_mask()  # [B, 1, H, W] 
datacube["mask"] = cloud_mask

# Generate embeddings with nodata handling
embeddings = model.embed(datacube)
```

Common nodata values are detected automatically: `NaN`, `-9999`, `0`, `-32768`. See [docs/nodata_handling.md](../nodata_handling.md) for details.

## Need Help?

- 📖 **Full Documentation**: [clay-foundation.github.io/model](https://clay-foundation.github.io/model)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Clay-foundation/model/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Clay-foundation/model/discussions)
