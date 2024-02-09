#!/usr/bin/env python3

# import sys
# sys.path.append("/home/tam/Documents/repos/model")

import os
import tempfile
from math import floor

import boto3
import einops
import geopandas as gpd
import numpy
import pyarrow as pa
import rasterio
import torch
from rasterio.windows import Window
from shapely import box
from torchvision.transforms import v2

from src.datamodule import ClayDataset
from src.model_clay import CLAYModule

YEAR = int(os.environ.get("YEAR", 2020))
DATE = f"{YEAR}-06-01"
TILE_SIZE = 12000
CHIP_SIZE = 512
E_W_INDEX_START = 67
E_W_INDEX_END = 125
N_S_INDEX_START = 24
N_S_INDEX_END = 49
YORIGIN = 50.0
XORIGIN = -125.0
PXSIZE = 8.333333333333333e-05

RASTER_X_SIZE = (E_W_INDEX_END - E_W_INDEX_START) * TILE_SIZE
RASTER_Y_SIZE = (N_S_INDEX_END - N_S_INDEX_START) * TILE_SIZE
NODATA = 0
# CKPT_PATH = "s3://clay-model-ckpt/v0/mae_epoch-24_val-loss-0.46.ckpt"
CKPT_PATH = "https://huggingface.co/made-with-clay/Clay/resolve/main/Clay_v0.1_epoch-24_val-loss-0.46.ckpt"
VERSION = "002"
BUCKET = "clay-worldcover-embeddings"
URL = "https://esa-worldcover-s2.s3.amazonaws.com/rgbnir/{year}/N{yidx}/ESA_WorldCover_10m_{year}_v{version}_N{yidx}W{xidx}_S2RGBNIR.tif"
WC_VERSION_LOOKUP = {
    2020: 100,
    2021: 200,
}


MEAN = [
    1369.03,  # red
    1597.68,  # green
    1741.10,  # blue
    2858.43,  # nir
]
STD = [
    2026.96,  # red
    2011.88,  # green
    2146.35,  # blue
    2016.38,  # nir
]

grid = gpd.read_file(
    # "/home/tam/Desktop/usa/esa_worldcover_grid_usa.fgb"
    "https://clay-mgrs-samples.s3.amazonaws.com/esa_worldcover_grid_usa.fgb"
)


def tiles_and_windows(input: Window):
    print("Input", input)
    x_tile_index = E_W_INDEX_END - floor(input.col_off / TILE_SIZE)
    x_local_off = input.col_off % TILE_SIZE
    x_size = min(CHIP_SIZE, TILE_SIZE - x_local_off)
    x_another = x_size < CHIP_SIZE

    y_tile_index = N_S_INDEX_END - floor(input.row_off / TILE_SIZE)
    y_local_off = input.row_off % TILE_SIZE
    y_size = min(CHIP_SIZE, TILE_SIZE - y_local_off)
    y_another = y_size < CHIP_SIZE

    tile_id = f"N{y_tile_index}W{str(x_tile_index).zfill(3)}"
    if tile_id not in grid.tile.values:
        return

    result = [
        (
            URL.format(
                yidx=y_tile_index,
                xidx=str(x_tile_index).zfill(3),
                year=YEAR,
                version=WC_VERSION_LOOKUP[YEAR],
            ),
            Window(x_local_off, y_local_off, x_size, y_size),
        )
    ]

    if x_another:
        result.append(
            (
                URL.format(
                    yidx=y_tile_index,
                    xidx=str(x_tile_index - 1).zfill(3),
                    year=YEAR,
                    version=WC_VERSION_LOOKUP[YEAR],
                ),
                Window(0, y_local_off, CHIP_SIZE - x_size, y_size),
            )
        )
    if y_another:
        result.append(
            (
                URL.format(
                    yidx=y_tile_index - 1,
                    xidx=str(x_tile_index).zfill(3),
                    year=YEAR,
                    version=WC_VERSION_LOOKUP[YEAR],
                ),
                Window(x_local_off, 0, x_size, CHIP_SIZE - y_size),
            )
        )
    if x_another and y_another:
        result.append(
            (
                URL.format(
                    yidx=y_tile_index - 1,
                    xidx=str(x_tile_index - 1).zfill(3),
                    year=YEAR,
                    version=WC_VERSION_LOOKUP[YEAR],
                ),
                Window(0, 0, CHIP_SIZE - x_size, CHIP_SIZE - y_size),
            )
        )

    return result


def make_batch(result):
    pixels = []
    for url, win in result:
        with rasterio.open(url) as src:
            data = src.read(window=win)
            if NODATA in data:
                return
            pixels.append(data)
            transform = src.window_transform(win)

    if len(pixels) == 1:
        pixels = pixels[0]
    elif len(pixels) == 2:  # noqa: PLR2004
        if pixels[0].shape[2] == CHIP_SIZE:
            pixels = einops.pack(pixels, "b * w")[0]
        else:
            pixels = einops.pack(pixels, "b h *")[0]
    else:
        px1 = einops.pack(pixels[:2], "b w *")[0]
        px2 = einops.pack(pixels[2:], "b w *")[0]
        pixels = einops.pack((px1, px2), "b * w")[0]

    assert pixels.shape == (4, CHIP_SIZE, CHIP_SIZE)

    return {
        "pixels": torch.as_tensor(data=[pixels], dtype=torch.float32).to(
            rgb_model.device
        ),
        "latlon": torch.as_tensor(
            data=[ds.normalize_latlon(transform[0], transform[3])]
        ).to(rgb_model.device),
        "timestep": torch.as_tensor(data=[ds.normalize_timestamp(f"{YEAR}-06-01")]).to(
            rgb_model.device
        ),
    }


index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 2))

# Setup model components
tfm = v2.Compose([v2.Normalize(mean=MEAN, std=STD)])
ds = ClayDataset(chips_path=[], transform=tfm)

rgb_model = CLAYModule.load_from_checkpoint(
    CKPT_PATH,
    mask_ratio=0.0,
    band_groups={"rgb": (0, 1, 2), "nir": (3,)},
    strict=False,  # ignore the extra parameters in the checkpoint
)

xoff = index * CHIP_SIZE
yoff = 0
embeddings = []
all_bounds = []
while yoff < RASTER_Y_SIZE:
    result = tiles_and_windows(Window(xoff, yoff, CHIP_SIZE, CHIP_SIZE))

    if result is None:
        yoff += CHIP_SIZE
        continue

    batch = make_batch(result)
    if batch is None:
        yoff += CHIP_SIZE
        continue

    (
        unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
    ) = rgb_model.model.encoder(batch)

    embeddings.append(unmasked_patches.detach().cpu().numpy())
    all_bounds.append(
        (
            XORIGIN + PXSIZE * xoff,
            YORIGIN - PXSIZE * (yoff + CHIP_SIZE),
            XORIGIN + PXSIZE * (xoff + CHIP_SIZE),
            YORIGIN - PXSIZE * yoff,
        )
    )

    yoff += CHIP_SIZE

embeddings = numpy.vstack(embeddings)

embeddings_mean = embeddings[:, :-2, :].mean(axis=1)

print(f"Average embeddings have shape {embeddings_mean.shape}")

gdf = gpd.GeoDataFrame(
    data={
        "embeddings": pa.FixedShapeTensorArray.from_numpy_ndarray(
            numpy.ascontiguousarray(embeddings_mean)
        ),
    },
    geometry=[box(*dat) for dat in all_bounds],  # This assumes same order
    crs="EPSG:4326",
)

with tempfile.TemporaryDirectory() as tmp:
    # tmp = "/home/tam/Desktop/wcctmp"

    outpath = f"{tmp}/worldcover_embeddings_{YEAR}_{index}_v{VERSION}.gpq"
    print(f"Uploading embeddings to {outpath}")

    gdf.to_parquet(path=outpath, compression="ZSTD", schema_version="1.0.0")

    s3_client = boto3.client("s3")
    s3_client.upload_file(
        outpath,
        BUCKET,
        f"v{VERSION}/{YEAR}/{os.path.basename(outpath)}",
    )
