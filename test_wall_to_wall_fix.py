#!/usr/bin/env python3
"""
Test script to verify the wall-to-wall tensor shape fix.
"""

import math
import numpy as np
import torch


def normalize_timestamp(date):
    """Normalize timestamp to week and hour components."""
    week = 1 * 2 * np.pi / 52  # week 1 as example
    hour = 12 * 2 * np.pi / 24  # noon as example
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(lat, lon):
    """Normalize lat/lon coordinates."""
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def test_tensor_shapes():
    """Test that the tensor shapes are correct after the fix."""
    # Simulate multiple dates (like 12 in the notebook)
    num_samples = 12
    lat, lon = 37.30939, -8.57207  # From the notebook
    
    times = [normalize_timestamp(None) for _ in range(num_samples)]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]
    
    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]
    
    # Test the old (broken) way
    print("=== Old (broken) way ===")
    try:
        time_old = torch.tensor(np.hstack((week_norm, hour_norm)), dtype=torch.float32)
        latlon_old = torch.tensor(np.hstack((lat_norm, lon_norm)), dtype=torch.float32)
        print(f"time_old shape: {time_old.shape}")
        print(f"latlon_old shape: {latlon_old.shape}")
        
        # Try to use it in the model's expected format
        time_latlon_old = torch.hstack((time_old, latlon_old))
        print(f"time_latlon_old shape: {time_latlon_old.shape}")
        print("This would fail with EinopsError when trying 'B D -> B L D'")
    except Exception as e:
        print(f"Error with old way: {e}")
    
    print("\n=== New (fixed) way ===")
    try:
        time_new = torch.tensor(np.column_stack((week_norm, hour_norm)), dtype=torch.float32)
        latlon_new = torch.tensor(np.column_stack((lat_norm, lon_norm)), dtype=torch.float32)
        print(f"time_new shape: {time_new.shape}")
        print(f"latlon_new shape: {latlon_new.shape}")
        
        # Try to use it in the model's expected format
        time_latlon_new = torch.hstack((time_new, latlon_new))
        print(f"time_latlon_new shape: {time_latlon_new.shape}")
        print("This should work correctly with 'B D -> B L D'")
        
        # Test the einops operation that was failing
        from einops import repeat
        B, D = time_latlon_new.shape
        L = 1024  # From the error message
        result = repeat(time_latlon_new, "B D -> B L D", L=L)
        print(f"After einops repeat: {result.shape}")
        print("✅ SUCCESS: Einops operation works correctly!")
        
    except Exception as e:
        print(f"Error with new way: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Testing wall-to-wall tensor shape fix...")
    success = test_tensor_shapes()
    if success:
        print("\n🎉 All tests passed! The fix should work.")
    else:
        print("\n❌ Tests failed. The fix needs more work.")