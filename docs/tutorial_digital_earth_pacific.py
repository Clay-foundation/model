# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Digital Earth Pacific mineral resource detection using Clay
#
# This notebook applies the Clay model on imagery composites, specifically a Sentinel-2
# Geometric Median [(GeoMAD)](https://docs.digitalearthafrica.org/en/latest/data_specs/GeoMAD_specs.html)
# and Sentinel-1 mean composite. We will use
# [Digital Earth Pacific's STAC API](https://stac-browser.staging.digitalearthpacific.org)
# to obtain these datasets, and apply it on a mineral resource detection downstream task
# on two levels:
#
# 1. Coarse level (chip of size 5.12km x 5.12km) - Using embeddings to get a general
# semantic picture
# 2. Fine level (pixel of size 10m x 10m) - Using a fine-tuned decoder head to get
# pixel-level segmentation masks
#
# References:
# - https://github.com/digitalearthpacific/mineral-resource-detection
# - https://github.com/Clay-foundation/model/discussions/140

# %%

import geopandas as gpd
import pystac_client
import shapely
import stackstac
from rasterio.enums import Resampling

BAND_GROUPS = {
    "rgb": ["B04", "B03", "B02"],
    "rededge": ["B05", "B06", "B07", "B8A"],
    "nir": ["B08"],
    "swir": ["B11", "B12"],
    "sar": ["mean_vv", "mean_vh"],
}

STAC_API = "https://stac.staging.digitalearthpacific.org"
COLLECTION = "dep_s2_geomad"

# %% [markdown]
# ## Find Sentinel-2 and Sentinel-1 composites stored as Cloud-Optimized GeoTIFFs
#
# Define spatiotemporal query

# %%
# Define area of interest
area_of_interest = shapely.box(xmin=177.2, ymin=-18.4, xmax=178.9, ymax=-17.2)

# Define temporal range
daterange: dict = ["2021-01-01T00:00:00Z", "2021-12-31T23:59:59Z"]

# %%
catalog = pystac_client.Client.open(url=STAC_API)

sen2_search = catalog.search(
    collections=[COLLECTION],
    datetime=daterange,
    intersects=area_of_interest,
    max_items=100,
)

items = sen2_search.get_all_items()

print(f"Found {len(items)} items")

# %% [markdown]
# ## Download the data
# Get the data into a numpy array and visualize the imagery. STAC browser URL is at
# https://stac-browser.staging.digitalearthpacific.org

# %%
# Extract coordinate system from first item
epsg = items[0].properties["proj:epsg"]

# Convert point from lon/lat to UTM projection
poidf = gpd.GeoDataFrame(crs="OGC:CRS84", geometry=[area_of_interest.centroid]).to_crs(
    epsg
)
geom = poidf.iloc[0].geometry

# Create bounds of the correct size, the model
# requires 512x512 pixels at 10m resolution.
bounds = (geom.x - 2560, geom.y - 2560, geom.x + 2560, geom.y + 2560)

# Retrieve the pixel values, for the bounding box in
# the target projection. In this example we use only
# the RGB and NIR band groups.
stack = stackstac.stack(
    items,
    bounds=bounds,
    snap_bounds=False,
    epsg=epsg,
    resolution=10,
    dtype="float32",
    rescale=False,
    fill_value=0,
    assets=BAND_GROUPS["rgb"] + BAND_GROUPS["nir"],
    resampling=Resampling.nearest,
)

stack = stack.compute()
assert stack.shape == (1, 4, 512, 512)

stack.sel(band=["B04", "B03", "B02"]).plot.imshow(
    row="time", rgb="band", vmin=0, vmax=2000
)
