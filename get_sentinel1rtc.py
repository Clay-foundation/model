"""
Script to retrieve Sentinel-1 Radiometrically Terrain Corrected (RTC) tiles.

Based on:
- https://gitlab.com/frontierdevelopmentlab/2022-us-sarchangedetection/deepslide/-/blob/main/notebooks/data_sentinel-1_rtc.ipynb
- https://zen3geo.readthedocs.io/en/v0.6.2/stacking.html
"""
import os

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import stackstac
import xarray as xr

# %%
# Uncomment the line below and set your Planetary Computer subscription key
# os.environ["PC_SDK_SUBSCRIPTION_KEY"] = "abcdefghijklmnopqrstuvwxyz123456"

# %%
# Set up spatiotemporal query to get Sentinel-1 RTC data from STAC API
query = dict(
    bbox=[99.8, -0.24, 100.07, 0.15],  # West, South, East, North
    datetime=["2022-01-25T00:00:00Z", "2022-03-25T23:59:59Z"],
    collections=["sentinel-1-rtc"],
)
catalog = pystac_client.Client.open(
    url="https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
search = catalog.search(**query)
item_collection = search.item_collection()
print(item_collection.items)


# %%
# Stack all Sentinel-1 scenes along the time dimension into an xarray.DataArray
da_sen1: xr.DataArray = stackstac.stack(
    items=item_collection,
    assets=["vh", "vv"],  # SAR polarizations
    epsg=32647,  # UTM Zone 47N
    resolution=20,  # Spatial resolution of 20 metres
    bounds_latlon=[99.933681, -0.009951, 100.065765, 0.147054],  # W, S, E, N
    xy_coords="center",  # pixel centroid coords instead of topleft corner
    resampling=rasterio.enums.Resampling.nearest,
    dtype=np.float32,
    fill_value=np.nan,
)

# %%
# To fix TypeError: Invalid value for attr 'spec'
da_sen1.attrs["spec"] = str(da_sen1.spec)

# To fix ValueError: unable to infer dtype on variable None
for key, val in da_sen1.coords.items():
    if val.dtype == "object":
        print("Deleting", key)
        da_sen1 = da_sen1.drop_vars(names=key)

# Create xarray.Dataset datacube with VH and VV channels from SAR
da_vh: xr.DataArray = da_sen1.sel(band="vh", drop=True).rename("vh")
da_vv: xr.DataArray = da_sen1.sel(band="vv", drop=True).rename("vv")
ds_sen1: xr.Dataset = xr.merge(objects=[da_vh, da_vv], join="override")

print(ds_sen1)

# %%
# Save to Zarr
ds_sen1.to_zarr(store := "data/sen1rtc.zarr")
print(f"Saved to {store}")
