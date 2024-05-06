import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import rioxarray  # noqa: F401
import stackstac
import torch
import yaml
from box import Box
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely import Point
from sklearn import decomposition
from stacchip.processors.prechip import normalize_timestamp
from torchvision.transforms import v2

from src.model_clay_v1 import ClayMAEModule

STAC_API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"


def get_stack(lat, lon, start, end, size, gsd):
    # Search the catalogue
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()

    # Reduce to one per date
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    print(f"Found {len(items)} items")

    # Extract coordinate system from first item
    epsg = items[0].properties["proj:epsg"]

    # Convert point into the image projection
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    coords = poidf.iloc[0].geometry.coords[0]

    # Create bounds in projection
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    # Retrieve the pixel values, for the bounding box in
    # the target projection. In this example we use only
    # the RGB and NIR band groups.
    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype="float32",
        rescale=False,
        fill_value=0,
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
    )

    print(f"Working with stack of size {stack.shape}")

    return stack.compute()


def plot_rgb(stack):
    stack.sel(band=["red", "green", "blue"]).plot.imshow(
        row="time", rgb="band", vmin=0, vmax=2000, col_wrap=6
    )
    plt.show()


def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def load_model(ckpt, device="cuda"):
    torch.set_default_device(device)

    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path="configs/metadata.yaml", shuffle=False, mask_ratio=0
    )
    model.eval()

    return model.to(device)


def prep_datacube(model, stack):
    # Ensure Sentinel-2 as platform for now
    if stack.coords["s2:product_type"].values != "S2MSI2A":
        raise ValueError("Currently on S2MSIL2A stacks are allowed for this script")

    platform = "sentinel-2-l2a"

    # Extract mean, std, and wavelengths from metadata
    metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))
    mean = []
    std = []
    waves = []
    for band in stack.band:
        mean.append(metadata[platform].bands.mean[str(band.values)])
        std.append(metadata[platform].bands.std[str(band.values)])
        waves.append(metadata[platform].bands.wavelength[str(band.values)])

    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )

    # Prep datetimes embedding
    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    # Prep lat/lon embedding
    latlons = [normalize_latlon(*poi)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Prep pixels
    pixels = torch.from_numpy(stack.data.astype(np.float32))
    pixels = transform(pixels)

    # Prepare additional information
    return {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=model.device,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=model.device
        ),
        "pixels": pixels.to(model.device),
        "gsd": torch.tensor(stack.gsd.values, device=model.device),
        "waves": torch.tensor(waves, device=model.device),
    }


def generate_embeddings(model, stack):
    datacube = prep_datacube(model, stack)

    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)

    # The first embedding is the class token, which is the
    # overall single embedding. We extract that for PCA below.
    return unmsk_patch[:, 0, :].cpu().numpy()


# Point over Monchique Portugal
poi = 37.30939, -8.57207

# Dates of a large forest fire
start = "2018-07-01"
end = "2018-09-01"

stack = get_stack(*poi, start, end, gsd=10, size=64)
plot_rgb(stack)

model = load_model(ckpt="/home/tam/Downloads/mae_v0.53_last.ckpt", device="cuda")

embeddings = generate_embeddings(model, stack)

pca = decomposition.PCA(n_components=1)
pca_result = pca.fit_transform(embeddings)

plt.xticks(rotation=-30)
# All points
plt.scatter(stack.time, pca_result, color="blue")

# Cloudy images
plt.scatter(stack.time[0], pca_result[0], color="green")
plt.scatter(stack.time[2], pca_result[2], color="green")

# After fire
plt.scatter(stack.time[-5:], pca_result[-5:], color="red")

plt.show()
