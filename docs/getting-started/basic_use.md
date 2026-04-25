# Basic Use

## Quick Start with Pretrained Model

The most common use case is generating embeddings with the pretrained Clay v1.5 model:

```python
import torch
from claymodel import load_metadata, normalize
from claymodel.api import load_model

# Load pretrained model (uses bundled metadata, sets mask_ratio=0)
model = load_model(ckpt_path="clay-v1.5.ckpt")

# Load sensor metadata (bundled with package — no file path needed)
metadata = load_metadata()

# Example: Generate embeddings for a Sentinel-2 chip
sensor = "sentinel-2-l2a"
sensor_meta = metadata[sensor]

# Wavelengths are in micrometers — use as-is from metadata
waves = torch.tensor(list(sensor_meta.bands.wavelength.values()))

# Your Sentinel-2 data: (batch, bands, height, width) = (1, 10, 256, 256)
pixels = torch.randn(1, 10, 256, 256)

# Normalize using per-band z-score statistics from training
pixels = normalize(pixels, sensor, metadata=metadata)

# Build datacube dict — the model's expected input format
datacube = {
    "pixels": pixels,
    "time": torch.zeros(1, 4),     # (week_sin, week_cos, hour_sin, hour_cos)
    "latlon": torch.zeros(1, 4),   # (lat_sin, lat_cos, lon_sin, lon_cos)
    "gsd": torch.tensor(float(sensor_meta.gsd)),
    "waves": waves,
}

# Generate 1024-dimensional embeddings
with torch.no_grad():
    encoded, *_ = model.encoder(datacube)
    embeddings = encoded[:, 0, :]  # CLS token

print(f"Generated embeddings shape: {embeddings.shape}")  # [1, 1024]
print(f"Using {sensor} with {len(waves)} bands at {sensor_meta.gsd}m resolution")
```

## Supported Sensors

Clay v1.5 is **sensor-agnostic** and can work with **any satellite instrument** as long as you provide the required metadata. The `configs/metadata.yaml` file contains specifications for commonly used sensors:

```python
from claymodel import load_metadata

# Load and display all supported sensors
metadata = load_metadata()

print("🛰️ CLAY v1.5 SUPPORTED SENSORS:")
print("=" * 60)

sensor_categories = {
    "Multispectral Satellites": ["sentinel-2-l2a", "landsat-c2l1", "landsat-c2l2-sr"],
    "Commercial High-Resolution": ["planetscope-sr"],
    "Aerial Imagery": ["naip", "linz"],
    "Radar": ["sentinel-1-rtc"],
    "Global Monitoring": ["modis"]
}

for category, sensors in sensor_categories.items():
    print(f"\n📡 {category}:")
    for sensor_name in sensors:
        if sensor_name in metadata:
            sensor_data = metadata[sensor_name]
            bands = sensor_data["band_order"]
            gsd = sensor_data["gsd"]
            num_bands = len(bands)
            print(f"   • {sensor_name}: {num_bands} bands, {gsd}m GSD")

print(f"\n🎯 Total supported sensors: {len(metadata)} (and growing!)")
```

## Adding New Sensors

Clay can work with **any satellite instrument**! To add a new sensor, simply add its specification to `configs/metadata.yaml`:

```yaml
# Example: Adding a new instrument
your-new-sensor:
  band_order:                    # List bands in the order they appear in your data
    - blue
    - green
    - red
    - nir
  rgb_indices: [2, 1, 0]        # Which bands to use for RGB visualization
  gsd: 10.0                     # Ground sampling distance in meters
  bands:
    mean:                       # Mean values for normalization (compute from your data)
      blue: 1200.0
      green: 1400.0
      red: 1600.0
      nir: 2800.0
    std:                        # Standard deviation for normalization
      blue: 400.0
      green: 450.0
      red: 500.0
      nir: 650.0
    wavelength:                 # Central wavelength in micrometers
      blue: 0.485
      green: 0.560
      red: 0.660
      nir: 0.835
```

### Computing Normalization Statistics

For new sensors, compute normalization statistics from your training data:

```python
import torch
import numpy as np

def compute_normalization_stats(data_chips, band_names):
    """
    Compute mean and std for each band across all chips.

    Args:
        data_chips: Tensor of shape [N, bands, height, width]
        band_names: List of band names
    """
    # Compute statistics across spatial and sample dimensions
    means = torch.mean(data_chips, dim=[0, 2, 3])  # Average over N, H, W
    stds = torch.std(data_chips, dim=[0, 2, 3])    # Std over N, H, W

    print("Normalization statistics for your sensor:")
    print("mean:")
    for i, band in enumerate(band_names):
        print(f"  {band}: {means[i]:.1f}")
    print("std:")
    for i, band in enumerate(band_names):
        print(f"  {band}: {stds[i]:.1f}")

# Example usage
# your_data = torch.randn(1000, 4, 256, 256)  # 1000 chips, 4 bands
# compute_normalization_stats(your_data, ["blue", "green", "red", "nir"])
```

### Contributing New Sensors

We welcome contributions of new sensor specifications! To contribute:

1. **Fork the repository** on GitHub
2. **Add your sensor** to `configs/metadata.yaml`
3. **Test your sensor** with Clay to ensure it works
4. **Submit a pull request** with:
   - Sensor metadata
   - Brief description of the instrument
   - Example usage (optional)

Popular sensors we'd love to see added:
- **VIIRS** (NOAA/NASA)
- **Hyperion** (hyperspectral)
- **CHRIS/PROBA** (hyperspectral)
- **RapidEye** (Planet)
- **SkySat** (Planet)
- **IKONOS** (Maxar)
- **GeoEye** (Maxar)
- **EROS** (ImageSat)

### Local Development with New Sensors

For local development, you can:

1. **Copy the metadata file** to your project:
   ```bash
   cp configs/metadata.yaml my_local_metadata.yaml
   ```

2. **Add your sensor** to the local copy

3. **Use your local metadata** in code:
   ```python
   with open("my_local_metadata.yaml", "r") as f:
       metadata = yaml.safe_load(f)
   ```

This approach lets you experiment with new sensors without modifying the main repository.

## Working with Different Sensors

Clay v1.5 supports multiple satellite sensors. Use the included metadata file for accurate wavelengths and normalization:

```python
import torch
from claymodel import load_metadata

# Load metadata for all supported sensors
metadata = load_metadata()

# Get wavelengths for any sensor — values are in micrometers (μm),
# which is the unit the model expects. Do NOT convert to nanometers.
def get_wavelengths(sensor_name):
    sensor_meta = metadata[sensor_name]
    return torch.tensor(list(sensor_meta.bands.wavelength.values()))

# Get wavelengths for different sensors
s2_waves = get_wavelengths("sentinel-2-l2a")       # 10 bands, 10m GSD
landsat_waves = get_wavelengths("landsat-c2l2-sr")  # 6 bands, 30m GSD
naip_waves = get_wavelengths("naip")                # 4 bands, 1m GSD
linz_waves = get_wavelengths("linz")                # 3 bands, 0.5m GSD
s1_waves = get_wavelengths("sentinel-1-rtc")        # 2 bands (synthetic wavelengths)
modis_waves = get_wavelengths("modis")              # 7 bands, 500m GSD

print(f"Sentinel-2 wavelengths (μm): {s2_waves}")
print(f"Landsat wavelengths (μm): {landsat_waves}")
print(f"NAIP wavelengths (μm): {naip_waves}")
```

## Data Normalization

Use the metadata file for proper data normalization:

```python
import torch
from claymodel import normalize

# The normalize() function applies per-band z-score normalization
# using the same statistics used during Clay v1.5 training.

# Example: Normalize Sentinel-2 data
raw_s2_chips = torch.randn(1, 10, 256, 256) * 2000 + 1500  # Simulated raw values
normalized_s2 = normalize(raw_s2_chips, "sentinel-2-l2a")

print(f"Raw range: {raw_s2_chips.min():.0f} to {raw_s2_chips.max():.0f}")
print(f"Normalized range: {normalized_s2.min():.2f} to {normalized_s2.max():.2f}")

# For Sentinel-1 SAR, convert to dB scale first:
# raw_sar = torch.randn(1, 2, 256, 256).abs()  # linear power
# sar_db = 10 * torch.log10(raw_sar.clamp(min=1e-10))
# normalized_sar = normalize(sar_db, "sentinel-1-rtc")
```

## Batch Processing

For processing multiple chips efficiently:

```python
import torch
from claymodel import load_metadata, normalize
from claymodel.api import load_model

model = load_model(ckpt_path="clay-v1.5.ckpt")
metadata = load_metadata()

# Process batch of Sentinel-2 chips
batch_size = 8
sensor = "sentinel-2-l2a"
sensor_meta = metadata[sensor]

# Simulated batch of chips — normalize before feeding to model
chips = torch.randn(batch_size, 10, 256, 256)
chips = normalize(chips, sensor, metadata=metadata)

datacube = {
    "pixels": chips,
    "time": torch.zeros(batch_size, 4),
    "latlon": torch.zeros(batch_size, 4),
    "gsd": torch.tensor(float(sensor_meta.gsd)),
    "waves": torch.tensor(list(sensor_meta.bands.wavelength.values())),
}

with torch.no_grad():
    encoded, *_ = model.encoder(datacube)
    embeddings = encoded[:, 0, :]  # CLS token per chip

print(f"Batch embeddings shape: {embeddings.shape}")  # [8, 1024]
```

## Complete Example: Multi-Sensor Processing

Here's a complete example showing how to process data from different sensors:

```python
import torch
from claymodel import load_metadata, normalize
from claymodel.api import load_model

# Load model and metadata
model = load_model(ckpt_path="clay-v1.5.ckpt")
metadata = load_metadata()

def process_sensor_data(pixels, sensor_name):
    """Process chips from any supported sensor."""
    sensor_meta = metadata[sensor_name]

    # Normalize and build datacube
    normalized = normalize(pixels, sensor_name, metadata=metadata)
    datacube = {
        "pixels": normalized,
        "time": torch.zeros(1, 4),
        "latlon": torch.zeros(1, 4),
        "gsd": torch.tensor(float(sensor_meta.gsd)),
        "waves": torch.tensor(list(sensor_meta.bands.wavelength.values())),
    }

    with torch.no_grad():
        encoded, *_ = model.encoder(datacube)
        return encoded[:, 0, :]  # CLS token

# Example with different sensors
sensors_to_test = ["sentinel-2-l2a", "naip", "landsat-c2l2-sr"]

for sensor in sensors_to_test:
    sensor_meta = metadata[sensor]
    num_bands = len(sensor_meta.band_order)

    # Simulate data for this sensor
    pixels = torch.randn(1, num_bands, 256, 256)
    embeddings = process_sensor_data(pixels, sensor)

    print(f"{sensor}: {num_bands} bands → {embeddings.shape[1]}D embedding")
```

## Running Jupyter Lab

If you installed the development environment:

    mamba activate claymodel
    python -m ipykernel install --user --name claymodel  # to install virtual env properly
    jupyter kernelspec list --json                       # see if kernel is installed
    jupyter lab &

## Training and Development

The neural network model can be trained via
[LightningCLI v2](https://pytorch-lightning.medium.com/introducing-lightningcli-v2supercharge-your-training-c070d43c7dd6).

> [!NOTE]
> For training, you'll need the full development environment with the repository cloned.

To check out the different options available, and look at the hyperparameter
configurations, run:

    python trainer.py --help

To quickly test the model on one batch in the validation set:

    python trainer.py fit --model ClayMAEModule --data ClayDataModule --config configs/config.yaml --trainer.fast_dev_run=True

To train the model:

    python trainer.py fit --model ClayMAEModule --data ClayDataModule --config configs/config.yaml

More options can be found using `python trainer.py fit --help`, or at the
[LightningCLI docs](https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html).

## Next Steps

- Try the [embeddings tutorial](../tutorials/embeddings.ipynb) for detailed examples
- Explore [reconstruction tutorial](../tutorials/reconstruction.ipynb) to see how the model works
- Check out [finetune examples](../finetune/classify.md) for downstream task training
