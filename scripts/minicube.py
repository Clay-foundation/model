from pathlib import Path

import geopandas as gpd
import pandas as pd
import pystac_client
import rasterio
import rioxarray  # noqa: F401
import stackstac
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely import Point

STAC_API = "https://earth-search.aws.element84.com/v1"
# STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"

COLLECTION = "sentinel-2-l2a"

BAND_GROUPS = {
    "rgb": ["red", "green", "blue"],
    "rededge": ["rededge1", "rededge2", "rededge3", "nir08"],
    "nir": [
        "nir",
    ],
    "swir": ["swir16", "swir22"],
    "sar": ["vv", "vh"],
}

# lat/lon
poi = 39.99146, -8.06256  # Pedrogao, Portugal
poi = 37.30939, -8.57207  # Monchique Portugal
# poi = 29.5672, 116.1346  # Poyang, China
# poi = 21.8978476,106.2495839  # Bac Son, Vietnam
# poi = 10.22651, 105.21669  # Mekong delta, Vietnam

start = "2018-07-01"
end = "2018-09-01"

catalog = pystac_client.Client.open(STAC_API)

search = catalog.search(
    collections=[COLLECTION],
    datetime=f"{start}/{end}",
    bbox=(poi[1] - 1e-5, poi[0] - 1e-5, poi[1] + 1e-5, poi[0] + 1e-5),
    max_items=100,
    query={"eo:cloud_cover": {"lt": 80}},
)

items = search.get_all_items()

print(f"Found {len(items)} items")

epsg = items[0].properties["proj:epsg"]

poidf = gpd.GeoDataFrame(
    pd.DataFrame(),
    crs="EPSG:4326",
    geometry=[Point(poi[1], poi[0])],
).to_crs(epsg)

coords = poidf.iloc[0].geometry.coords[0]

bounds = (
    coords[0] - 2560,
    coords[1] - 2560,
    coords[0] + 2560,
    coords[1] + 2560,
)


stack = stackstac.stack(
    items,
    bounds=bounds,
    snap_bounds=False,
    epsg=epsg,
    resolution=10,
    dtype="float32",
    rescale=False,
    fill_value=0,
    assets=BAND_GROUPS["rgb"],
    resampling=Resampling.bilinear,
)

stack = stack.compute()

stack.plot.imshow(row="time", rgb="band", vmin=0, vmax=2000, col_wrap=7)
plt.show()

outdir = Path("data/minicubes")
assert outdir.exists()

# Write tile to output dir
for tile in stack:
    # Grid code like MGRS-29SNB
    mgrs = str(tile.coords["grid:code"].values).split("-")[1]
    date = str(tile.time.values)[:10]

    name = "{dir}/claytile_{mgrs}_{date}.tif".format(
        dir=outdir,
        mgrs=mgrs,
        date=date.replace("-", ""),
    )
    tile.rio.to_raster(name, compress="deflate")

    with rasterio.open(name, "r+") as rst:
        rst.update_tags(date=date)
