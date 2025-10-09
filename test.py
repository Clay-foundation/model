import yaml
import torch
# print('works')

from claymodel.module import ClayMAEModule

# Load pretrained model
model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
model.eval()

with open("configs/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)


# Example: Generate embeddings for a Sentinel-2 chip
sensor = "sentinel-2-l2a"
sensor_meta = metadata[sensor]

# print(sensor_meta)

# Get wavelengths from metadata (convert from μm to nm)

wavelengths = []
for band in sensor_meta["band_order"]:
    wavelength_nm = sensor_meta["bands"]["wavelength"][band] * 1000
    wavelengths.append(wavelength_nm)
print(wavelengths)
wavelengths = torch.tensor([wavelengths], dtype=torch.float32)

print(wavelengths)

# Your Sentinel-2 data: (batch, bands, height, width) = (1, 10, 256, 256)

chips = torch.randn(1, 10, 256, 256)
timestamps = torch.tensor([0, 0, 0, 0], dtype=torch.float32)  # [week, hour, lat, lon]

# Generate 1024-dimensional embeddings

with torch.no_grad():
    embeddings = model.encoder(chips, timestamps, wavelengths)

print(f"Generated embeddings shape: {embeddings.shape}")  # [1, 1024]
print(
    f"Using {sensor} with {len(wavelengths[0])} bands at {sensor_meta['gsd']}m resolution"
)
