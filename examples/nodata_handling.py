"""
Example of how to handle nodata pixels with Clay model.

This example demonstrates how to:
1. Create masks for nodata pixels
2. Run inference with masked data
3. Handle different nodata scenarios
"""

import torch
import numpy as np
from claymodel.utils import create_nodata_mask, create_datacube_with_mask
from claymodel.module import ClayMAEModule

def example_basic_nodata_handling():
    """Basic example of creating and using nodata masks."""
    
    print("=== Basic Nodata Handling Example ===")
    
    # Create example data with some nodata values
    batch_size, channels, height, width = 1, 4, 256, 256
    pixels = torch.randn(batch_size, channels, height, width)
    
    # Introduce some nodata values
    pixels[0, :, 50:100, 50:100] = -9999  # Common nodata value
    pixels[0, :, 150:200, 150:200] = 0     # Zero as nodata
    pixels[0, 0, 200:250, 200:250] = float('nan')  # NaN values
    
    print(f"Original data shape: {pixels.shape}")
    print(f"Number of -9999 values: {(pixels == -9999).sum()}")
    print(f"Number of zero values: {(pixels == 0).sum()}")
    print(f"Number of NaN values: {torch.isnan(pixels).sum()}")
    
    # Create nodata mask
    mask = create_nodata_mask(pixels)
    print(f"Mask shape: {mask.shape}")
    print(f"Total masked pixels: {mask.sum()}")
    print(f"Percentage masked: {mask.float().mean() * 100:.1f}%")
    
    # Create time and location tensors
    time = torch.zeros(batch_size, 4)  # [week_norm, hour_norm, lat, lon] 
    latlon = torch.zeros(batch_size, 4)
    platform = "sentinel-2-l2a"
    
    # Create datacube with mask
    datacube = create_datacube_with_mask(
        pixels=pixels,
        time=time, 
        latlon=latlon,
        platform=platform,
        mask=mask
    )
    
    print(f"Datacube keys: {list(datacube.keys())}")
    return datacube


def example_platform_specific_nodata():
    """Example showing platform-specific nodata handling."""
    
    print("\n=== Platform-Specific Nodata Handling ===")
    
    # Sentinel-1 example with negative/zero values
    s1_pixels = torch.randn(1, 2, 256, 256)
    s1_pixels[s1_pixels < 0] = -32768  # Sentinel-1 common nodata
    s1_pixels[0, :, 100:150, 100:150] = 0  # Zero backscatter (nodata)
    
    s1_mask = create_nodata_mask(s1_pixels, platform="sentinel-1-rtc") 
    print(f"Sentinel-1 mask covers {s1_mask.float().mean() * 100:.1f}% of pixels")
    
    # Sentinel-2 example with cloud masks
    s2_pixels = torch.randn(1, 10, 256, 256) * 1000 + 2000  # Typical reflectance values
    s2_pixels[0, :, 0:50, :] = -9999  # Typical Sentinel-2 nodata
    
    s2_mask = create_nodata_mask(s2_pixels, platform="sentinel-2-l2a")
    print(f"Sentinel-2 mask covers {s2_mask.float().mean() * 100:.1f}% of pixels")


def example_inference_with_nodata():
    """Example of running inference with nodata pixels."""
    
    print("\n=== Inference with Nodata Example ===")
    
    # Note: This requires a trained model checkpoint
    # For demonstration, we show the data preparation steps
    
    # Create sample data
    datacube = example_basic_nodata_handling()
    
    print("Ready for model inference!")
    print("The model will:")
    print("1. Convert pixel-level masks to patch-level masks") 
    print("2. Prioritize nodata patches for masking during training")
    print("3. Exclude nodata regions from loss computation")
    print("4. Generate embeddings that account for missing data")
    
    # Example of how you would run inference:
    # model = ClayMAEModule.load_from_checkpoint("path/to/checkpoint.ckpt")
    # model.eval()
    # with torch.no_grad():
    #     loss, recon_loss, repr_loss = model(datacube)
    #     print(f"Reconstruction loss: {recon_loss}")


def example_cloud_mask_integration():
    """Example of integrating external cloud masks."""
    
    print("\n=== Cloud Mask Integration Example ===")
    
    # Simulate loading external cloud mask (e.g., from scene classification)
    height, width = 256, 256
    cloud_mask = np.zeros((height, width), dtype=bool)
    
    # Add some cloud areas
    cloud_mask[50:150, 100:200] = True  # Large cloud area
    cloud_mask[200:230, 50:100] = True   # Smaller cloud area
    
    # Convert to tensor and proper shape [B, C, H, W] 
    cloud_mask_tensor = torch.from_numpy(cloud_mask).float()
    cloud_mask_tensor = cloud_mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Create pixel data
    pixels = torch.randn(1, 4, height, width) * 1000 + 2000
    
    # Combine with pixel-based nodata detection
    pixel_mask = create_nodata_mask(pixels)
    
    # Combine both masks
    combined_mask = torch.logical_or(
        pixel_mask,
        cloud_mask_tensor.expand_as(pixel_mask)
    ).float()
    
    print(f"Cloud pixels: {cloud_mask_tensor.sum()}")
    print(f"Pixel nodata: {pixel_mask.sum()}")  
    print(f"Combined mask: {combined_mask.sum()}")
    
    # Create datacube with combined mask
    time = torch.zeros(1, 4)
    latlon = torch.zeros(1, 4)
    
    datacube = create_datacube_with_mask(
        pixels=pixels,
        time=time,
        latlon=latlon, 
        platform="sentinel-2-l2a",
        mask=combined_mask
    )
    
    return datacube


if __name__ == "__main__":
    # Run all examples
    example_basic_nodata_handling()
    example_platform_specific_nodata()
    example_inference_with_nodata()
    example_cloud_mask_integration()
    
    print("\n=== Summary ===")
    print("✓ Nodata pixels can be handled through pixel-level masks")
    print("✓ Masks are automatically converted to patch-level for the model")
    print("✓ Common nodata values (-9999, 0, NaN) are detected automatically") 
    print("✓ Platform-specific handling is available")
    print("✓ External masks (e.g., cloud masks) can be integrated")
    print("✓ Backward compatibility maintained for data without masks")