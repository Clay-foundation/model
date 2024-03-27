from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import pystac_client
import rasterio
import rioxarray  # noqa: F401
import stackstac
import torch
from rasterio.enums import Resampling
from shapely import Point
from sklearn import decomposition

from src.datamodule import ClayDataModule
from src.model_clay import CLAYModule

# ###########################################################
# Download a time series of image chips over point location
# ###########################################################

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
    assets=BAND_GROUPS["rgb"] + BAND_GROUPS["nir"],
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


# ###########################################################
# Now switch gears and load the tiles to create embeddings
# and analyze them. Most of the partial input code is copied
# from the partial inputs notebook here
# https://github.com/Clay-foundation/model/blob/docs/model/docs/clay-v0-partial-inputs.ipynb
# ###########################################################

DATA_DIR = "data/minicubes"  # data directory for all chips
CKPT_PATH = "data/checkpoints/Clay_v0.1_epoch-24_val-loss-0.46.ckpt"


rgb_model = CLAYModule.load_from_checkpoint(
    CKPT_PATH,
    mask_ratio=0.0,  # mask out 70% of the input patches
    # bands=3,
    # band_groups={"rgb": (2, 1, 0)},
    # bands=4,
    band_groups={"rgb": (2, 1, 0), "nir": (3,)},
    strict=False,  # ignore the extra parameters in the checkpoint
)
rgb_model.eval()  # set the model to evaluation mode


class ClayDataModuleRGB(ClayDataModule):
    MEAN = [
        1369.03,
        1597.68,
        1741.10,
        2858.43,  # nir
    ]
    STD = [
        2026.96,
        2011.88,
        2146.35,
        2016.38,  # nir
    ]


data_dir = Path(DATA_DIR)

dm = ClayDataModuleRGB(data_dir=str(data_dir), batch_size=8)
dm.setup(stage="predict")
trn_dl = iter(dm.predict_dataloader())

embeddings = []

for batch in trn_dl:
    with torch.no_grad():
        # Move data from to the device of model
        batch["pixels"] = batch["pixels"].to(rgb_model.device)
        # Pass just the specific band through the model
        batch["timestep"] = batch["timestep"].to(rgb_model.device)
        batch["latlon"] = batch["latlon"].to(rgb_model.device)

        # Pass pixels, latlon, timestep through the encoder to create encoded patches
        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = rgb_model.model.encoder(batch)

        embeddings.append(unmasked_patches.detach().cpu().numpy())

embeddings = numpy.vstack(embeddings)

embeddings_mean = embeddings[:, :-2, :].mean(axis=1)

pca = decomposition.PCA(2)

# pca_result = pca.fit_transform(rearrange(embeddings, "o s w -> o (s w)"))
pca_result = pca.fit_transform(embeddings_mean)

plt.scatter(range(len(pca_result)), pca_result[:, 0], color="blue")
plt.scatter(range(len(pca_result)), pca_result[:, 1], color="red")
# plt.scatter(range(len(pca_result)), pca_result[:, 2], color="green")
# plt.scatter(range(len(pca_result)), pca_result)

plt.show()

# Load the RGB Clay Dataset
# ds = ClayDataset(chips_path=list(data_dir.glob("**/*.tif")))
# sample = ds[2]  # pick a random sample
# bgr = rearrange(sample["pixels"].cpu().numpy(), "c h w -> h w c")
# rgb = bgr[..., ::-1]  # reverse the order of the channels
# plt.imshow(rgb / 2000)
