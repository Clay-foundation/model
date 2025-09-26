"""
Code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

"""

import torch
import numpy as np


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d_with_gsd(
    h, w, dim, gsd=1.0, temperature: int = 10000, dtype=torch.float32
):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    gsd = gsd.to(x.device)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)  # Adjusted for g

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(waves, dim, temperature: int = 10000, dtype=torch.float32):
    assert (
        dim % 2 == 0
    ), "Feature dimension must be a multiple of 2 for sincos embedding"
    waves = torch.arange(waves) if isinstance(waves, int) else waves

    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_waves = waves[:, None] * omega[None, :]
    pe = torch.cat((scaled_waves.sin(), scaled_waves.cos()), dim=1)

    return pe.type(dtype)


def create_nodata_mask(pixels, nodata_values=None, platform=None):
    """
    Create a nodata mask from pixel values.
    
    Parameters
    ----------
    pixels : np.ndarray or torch.Tensor
        Input pixel values of shape [C, H, W] or [B, C, H, W]
    nodata_values : list, optional
        List of values to consider as nodata. If None, uses common defaults.
    platform : str, optional 
        Platform name to use platform-specific nodata handling.
        
    Returns
    -------
    mask : np.ndarray or torch.Tensor
        Boolean mask where True indicates nodata pixels. Same type as input.
        Shape [C, H, W] or [B, C, H, W] matching input.
    """
    is_torch = torch.is_tensor(pixels)
    
    if nodata_values is None:
        # Common nodata values across different platforms
        nodata_values = [0, -9999, -32768, 65535]
        
        # Platform-specific adjustments
        if platform == "sentinel-1-rtc":
            nodata_values.extend([float('-inf'), float('inf')])
    
    # Initialize mask
    if is_torch:
        mask = torch.zeros_like(pixels, dtype=torch.bool)
        # Check for NaN values
        mask = mask | torch.isnan(pixels)
        # Check for infinite values  
        mask = mask | torch.isinf(pixels)
    else:
        mask = np.zeros_like(pixels, dtype=bool)
        # Check for NaN values
        mask = mask | np.isnan(pixels)
        # Check for infinite values
        mask = mask | np.isinf(pixels)
    
    # Check each nodata value
    for nodata_val in nodata_values:
        if is_torch:
            mask = mask | (pixels == nodata_val)
        else:
            mask = mask | (pixels == nodata_val)
    
    return mask


def create_datacube_with_mask(pixels, time, latlon, platform, mask=None, nodata_values=None):
    """
    Create a datacube dictionary with optional nodata mask.
    
    Parameters
    ----------
    pixels : torch.Tensor
        Pixel data of shape [B, C, H, W]
    time : torch.Tensor 
        Time encoding of shape [B, 2] or [B, 4]
    latlon : torch.Tensor
        Location encoding of shape [B, 2] or [B, 4] 
    platform : str
        Platform name
    mask : torch.Tensor, optional
        Pre-computed mask of shape [B, C, H, W] or [B, 1, H, W]
    nodata_values : list, optional
        Values to treat as nodata if mask is not provided
        
    Returns
    -------
    datacube : dict
        Dictionary with keys: pixels, time, latlon, platform, mask
    """
    if mask is None:
        # Create mask from nodata values
        mask = create_nodata_mask(pixels, nodata_values, platform)
        
        # Convert to tensor if needed and ensure proper shape
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
            
        # If mask has same channels as pixels, reduce to single channel
        if mask.shape[1] == pixels.shape[1] and pixels.shape[1] > 1:
            mask = mask.any(dim=1, keepdim=True).float()
        elif len(mask.shape) == 4 and mask.shape[1] > 1:
            mask = mask.any(dim=1, keepdim=True).float()
    
    # Ensure mask is float32 for processing
    if mask.dtype != torch.float32:
        mask = mask.float()
    
    return {
        "pixels": pixels,
        "time": time, 
        "latlon": latlon,
        "platform": platform,
        "mask": mask,
    }
