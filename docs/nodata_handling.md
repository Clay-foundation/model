# Handling Nodata Pixels in Clay Model

This document describes how to handle nodata pixels (missing data due to projection effects, cloud masks, sensor issues, etc.) when using the Clay model for inference.

## Overview

The Clay model now supports automatic handling of nodata pixels through:

1. **Pixel-level masks**: Boolean masks indicating which pixels contain valid data
2. **Automatic nodata detection**: Common nodata values are detected automatically
3. **Patch-level masking**: Pixel masks are converted to patch-level masks for the Vision Transformer
4. **Loss computation**: Nodata regions are excluded from loss calculations during training
5. **Backward compatibility**: Existing code continues to work without masks

## Quick Start

```python
import torch
from claymodel.utils import create_datacube_with_mask
from claymodel.module import ClayMAEModule

# Your pixel data with some nodata values
pixels = torch.randn(1, 4, 256, 256)
pixels[0, :, 50:100, 50:100] = -9999  # Nodata region

# Create datacube with automatic nodata detection
datacube = create_datacube_with_mask(
    pixels=pixels,
    time=torch.zeros(1, 4),
    latlon=torch.zeros(1, 4), 
    platform="sentinel-2-l2a"
)

# Run inference
model = ClayMAEModule.load_from_checkpoint("checkpoint.ckpt")
with torch.no_grad():
    loss, recon_loss, repr_loss = model(datacube)
```

## Supported Nodata Values

The following values are automatically detected as nodata:

- `NaN` (Not a Number)
- `±inf` (Positive/negative infinity)
- `0` (Common for optical sensors)
- `-9999` (Standard nodata value)
- `-32768` (Common for integer data)
- `65535` (Max uint16, sometimes used as nodata)

Platform-specific extensions:
- **Sentinel-1**: Additionally detects negative values and very large values

## Creating Masks

### Automatic Detection

```python
from claymodel.utils import create_nodata_mask

# Automatic detection of common nodata values
mask = create_nodata_mask(pixels)

# Platform-specific detection
mask = create_nodata_mask(pixels, platform="sentinel-1-rtc")

# Custom nodata values
mask = create_nodata_mask(pixels, nodata_values=[0, -9999, 32767])
```

### Manual Mask Creation

```python
import torch

# Create custom mask (True = nodata, False = valid data)
mask = torch.zeros(1, 1, 256, 256, dtype=torch.bool)
mask[0, 0, 50:100, 50:100] = True  # Mark region as nodata
```

### External Mask Integration

```python
# Integrate cloud masks, quality flags, etc.
cloud_mask = load_cloud_mask()  # Your cloud mask loading function
quality_mask = load_quality_flags()  # Your quality mask

# Combine masks
combined_mask = torch.logical_or(
    torch.logical_or(cloud_mask, quality_mask),
    create_nodata_mask(pixels)
)

datacube = create_datacube_with_mask(
    pixels=pixels,
    time=time,
    latlon=latlon,
    platform=platform,
    mask=combined_mask
)
```

## How It Works

### 1. Pixel to Patch Conversion

The model converts pixel-level masks to patch-level masks:

- Each image is divided into patches (typically 32x32 pixels)
- A patch is considered "nodata" if more than 50% of its pixels are marked as nodata
- This patch-level mask guides the model's attention mechanism

### 2. Masking Strategy

During training and inference:

- **Nodata patches** are prioritized for masking (they get masked first)
- **Loss computation** excludes nodata patches from reconstruction loss
- **Embeddings** are generated considering the missing data context

### 3. Backward Compatibility

- Existing datacubes without masks continue to work unchanged
- If no mask is provided, an empty mask (all pixels valid) is created automatically
- All model outputs remain the same format

## Best Practices

### 1. Conservative Masking
Mark pixels as nodata conservatively. It's better to include slightly uncertain pixels than to mask too aggressively.

### 2. Quality Flag Integration
Combine nodata detection with quality flags from your data provider:

```python
# Example for Sentinel-2
scene_classification = load_scene_classification()  # SCL band
clouds_and_shadows = (scene_classification == 3) | (scene_classification == 8)  # Clouds and shadows

combined_mask = torch.logical_or(
    create_nodata_mask(pixels), 
    clouds_and_shadows
)
```

### 3. Consistent Processing
Apply the same nodata handling logic to both training and inference data for best results.

### 4. Patch Size Considerations
Since masking works at the patch level (32x32 pixels), small nodata regions might not be effectively masked. Consider the scale of your nodata regions relative to the patch size.

## API Reference

### `create_nodata_mask(pixels, nodata_values=None, platform=None)`

Create a boolean mask identifying nodata pixels.

**Parameters:**
- `pixels`: Input pixel data (torch.Tensor or numpy.ndarray)
- `nodata_values`: List of values to consider as nodata (optional)
- `platform`: Platform name for platform-specific handling (optional)

**Returns:**
- Boolean mask where True indicates nodata pixels

### `create_datacube_with_mask(pixels, time, latlon, platform, mask=None, nodata_values=None)`

Create a datacube dictionary with nodata mask support.

**Parameters:**
- `pixels`: Pixel data tensor [B, C, H, W]
- `time`: Time encoding tensor [B, 4]
- `latlon`: Location encoding tensor [B, 4]  
- `platform`: Platform name string
- `mask`: Pre-computed mask (optional)
- `nodata_values`: Custom nodata values (optional)

**Returns:**
- Dictionary with keys: pixels, time, latlon, platform, mask

## Examples

See `examples/nodata_handling.py` for complete working examples including:

- Basic nodata detection and masking
- Platform-specific handling
- Cloud mask integration
- Inference with masked data

## Troubleshooting

### High Mask Ratio
If your data has a very high percentage of nodata pixels (>80%), consider:
- Checking if your nodata detection is too aggressive
- Using different imagery with less cloud cover
- Adjusting your area of interest

### Performance Impact
The nodata handling adds minimal computational overhead:
- Mask computation is done once during data loading
- Patch-level masking is efficiently integrated into the existing masking mechanism
- No impact on model inference speed

### Memory Usage
Masks add approximately 1-4 bytes per pixel (depending on data type), which is typically <1% of total memory usage.