# Basic Use

## Quick Start with Pretrained Model

The most common use case is generating embeddings with the pretrained Clay v1.5 model:

```python
import yaml
import torch
from claymodel.module import ClayMAEModule

# Load pretrained model
model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
model.eval()

# Load sensor metadata
with open("configs/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)

# Example: Generate embeddings for a Sentinel-2 chip
sensor = "sentinel-2-l2a"
sensor_meta = metadata[sensor]

# Get wavelengths from metadata (convert from μm to nm)
wavelengths = []
for band in sensor_meta["band_order"]:
    wavelength_nm = sensor_meta["bands"]["wavelength"][band] * 1000
    wavelengths.append(wavelength_nm)
wavelengths = torch.tensor([wavelengths], dtype=torch.float32)

# Your Sentinel-2 data: (batch, bands, height, width) = (1, 10, 256, 256)
chips = torch.randn(1, 10, 256, 256)  
timestamps = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)  # [week, hour, lat, lon]

# Generate 1024-dimensional embeddings
with torch.no_grad():
    embeddings = model.encoder(chips, timestamps, wavelengths)

print(f"Generated embeddings shape: {embeddings.shape}")  # [1, 1024]
print(f"Using {sensor} with {len(wavelengths[0])} bands at {sensor_meta['gsd']}m resolution")
```

## Supported Sensors

Clay v1.5 is **sensor-agnostic** and can work with **any satellite instrument** as long as you provide the required metadata. The `configs/metadata.yaml` file contains specifications for commonly used sensors:

```python
import yaml

# Load and display all supported sensors
with open("configs/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)

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
import yaml
import torch
from claymodel.module import ClayMAEModule

# Load metadata for all supported sensors
with open("configs/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)

# Function to get wavelengths for any sensor
def get_wavelengths(sensor_name):
    sensor_meta = metadata[sensor_name]
    wavelengths = []
    for band in sensor_meta["band_order"]:
        # Convert from micrometers to nanometers (multiply by 1000)
        wavelength_nm = sensor_meta["bands"]["wavelength"][band] * 1000
        wavelengths.append(wavelength_nm)
    return torch.tensor([wavelengths], dtype=torch.float32)

# Get wavelengths for different sensors
s2_wavelengths = get_wavelengths("sentinel-2-l2a")      # 10 bands, 10m GSD
landsat_wavelengths = get_wavelengths("landsat-c2l2-sr") # 6 bands, 30m GSD  
naip_wavelengths = get_wavelengths("naip")              # 4 bands, 1m GSD
linz_wavelengths = get_wavelengths("linz")              # 3 bands, 0.5m GSD
s1_wavelengths = get_wavelengths("sentinel-1-rtc")     # 2 bands, 10m GSD
modis_wavelengths = get_wavelengths("modis")           # 7 bands, 500m GSD

print(f"Sentinel-2 wavelengths: {s2_wavelengths}")
print(f"Landsat wavelengths: {landsat_wavelengths}")
print(f"NAIP wavelengths: {naip_wavelengths}")
```

## Data Normalization

Use the metadata file for proper data normalization:

```python
import yaml
import torch

# Load metadata
with open("configs/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)

def normalize_data(chips, sensor_name):
    """Normalize chips using sensor-specific statistics from metadata."""
    sensor_meta = metadata[sensor_name]["bands"]
    
    # Get means and stds in band order
    means = torch.tensor([sensor_meta["mean"][band] for band in metadata[sensor_name]["band_order"]])
    stds = torch.tensor([sensor_meta["std"][band] for band in metadata[sensor_name]["band_order"]])
    
    # Normalize: (x - mean) / std
    # Reshape for broadcasting: [1, bands, 1, 1]
    means = means.view(1, -1, 1, 1)
    stds = stds.view(1, -1, 1, 1)
    
    normalized = (chips - means) / stds
    return normalized

# Example: Normalize Sentinel-2 data
raw_s2_chips = torch.randn(1, 10, 256, 256) * 2000 + 1500  # Simulated raw values
normalized_s2 = normalize_data(raw_s2_chips, "sentinel-2-l2a")

print(f"Raw range: {raw_s2_chips.min():.0f} to {raw_s2_chips.max():.0f}")
print(f"Normalized range: {normalized_s2.min():.2f} to {normalized_s2.max():.2f}")
```

## Batch Processing

For processing multiple chips efficiently:

```python
import yaml
import torch
from claymodel.module import ClayMAEModule

# Load metadata
with open("configs/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)

model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
model.eval()

# Process batch of Sentinel-2 chips
batch_size = 8
sensor = "sentinel-2-l2a"

# Get wavelengths from metadata
wavelengths = []
for band in metadata[sensor]["band_order"]:
    wavelengths.append(metadata[sensor]["bands"]["wavelength"][band] * 1000)  # Convert to nm
wavelengths = torch.tensor([wavelengths] * batch_size, dtype=torch.float32)

# Simulated batch of chips
chips = torch.randn(batch_size, 10, 256, 256)
timestamps = torch.zeros(batch_size, 4)  # [week, hour, lat, lon]

with torch.no_grad():
    embeddings = model.encoder(chips, timestamps, wavelengths)

print(f"Batch embeddings shape: {embeddings.shape}")  # [8, 1024]
```

## Complete Example: Multi-Sensor Processing

Here's a complete example showing how to process data from different sensors:

```python
import yaml
import torch
from claymodel.module import ClayMAEModule

# Load model and metadata
model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
model.eval()

with open("configs/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)

def process_sensor_data(chips, sensor_name):
    """Process chips from any supported sensor."""
    sensor_meta = metadata[sensor_name]
    
    # Get wavelengths
    wavelengths = []
    for band in sensor_meta["band_order"]:
        wavelengths.append(sensor_meta["bands"]["wavelength"][band] * 1000)
    wavelengths = torch.tensor([wavelengths], dtype=torch.float32)
    
    # Normalize data
    means = torch.tensor([sensor_meta["bands"]["mean"][band] for band in sensor_meta["band_order"]])
    stds = torch.tensor([sensor_meta["bands"]["std"][band] for band in sensor_meta["band_order"]])
    means = means.view(1, -1, 1, 1)
    stds = stds.view(1, -1, 1, 1)
    normalized_chips = (chips - means) / stds
    
    # Generate embeddings
    timestamps = torch.zeros(1, 4)  # Can be zeros if unknown
    with torch.no_grad():
        embeddings = model.encoder(normalized_chips, timestamps, wavelengths)
    
    return embeddings

# Example with different sensors
sensors_to_test = ["sentinel-2-l2a", "naip", "landsat-c2l2-sr"]

for sensor in sensors_to_test:
    sensor_meta = metadata[sensor]
    num_bands = len(sensor_meta["band_order"])
    
    # Simulate data for this sensor
    chips = torch.randn(1, num_bands, 256, 256)
    embeddings = process_sensor_data(chips, sensor)
    
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
