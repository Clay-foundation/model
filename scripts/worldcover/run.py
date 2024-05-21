#!/usr/bin/env python3

import sys

sys.path.append("../../")

import os
import tempfile
from math import floor
from pathlib import Path

import boto3
import einops
import geopandas as gpd
import numpy
import pandas as pd
import rasterio
import requests
import shapely
import torch
import xarray as xr
from rasterio.windows import Window
from torchvision.transforms import v2

from src.datamodule import ClayDataset
from src.model_clay import CLAYModule

YEAR = int(os.environ.get("YEAR", 2020))
DATE = f"{YEAR}-06-01"
TILE_SIZE = 12000
PATCH_SIZE = 32
CHIP_SIZE = 512
E_W_INDEX_START = 67
E_W_INDEX_END = 125
N_S_INDEX_START = 24
N_S_INDEX_END = 49
YORIGIN = 50.0
XORIGIN = -125.0
PXSIZE = 8.333333333333333e-05
SUCCESS_CODE = 200

RASTER_X_SIZE = (E_W_INDEX_END - E_W_INDEX_START) * TILE_SIZE
RASTER_Y_SIZE = (N_S_INDEX_END - N_S_INDEX_START) * TILE_SIZE
NODATA = 0
CKPT_PATH = (
    "https://huggingface.co/made-with-clay/Clay/resolve/main/"
    "Clay_v0.1_epoch-24_val-loss-0.46.ckpt"
)
VERSION = "005"
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


def download_image(url):
    # Download an image from a URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == SUCCESS_CODE:
        return response.content  # Return the image content
    else:
        raise Exception("Failed to download the image")


def patch_bounds_from_url(url, chunk_size=(PATCH_SIZE, PATCH_SIZE)):
    # Download an image from a URL
    image_data = download_image(url)

    # Open the image using rasterio from memory
    with rasterio.io.MemoryFile(image_data) as memfile:
        with memfile.open() as src:
            # Read the image data and metadata
            img_data = src.read()
            img_meta = src.profile
            img_crs = src.crs

    # Convert raster data and metadata into an xarray DataArray
    img_da = xr.DataArray(img_data, dims=("band", "y", "x"), attrs=img_meta)

    # Tile the data
    ds_chunked = img_da.chunk({"y": chunk_size[0], "x": chunk_size[1]})

    # Get the geospatial information from the original dataset
    transform = img_meta["transform"]

    # Iterate over the chunks and compute the geospatial bounds for each chunk
    chunk_bounds = {}

    for x in range(ds_chunked.sizes["x"] // chunk_size[1]):
        for y in range(ds_chunked.sizes["y"] // chunk_size[0]):
            # Compute chunk coordinates
            x_start = x * chunk_size[1]
            y_start = y * chunk_size[0]
            x_end = min(x_start + chunk_size[1], ds_chunked.sizes["x"])
            y_end = min(y_start + chunk_size[0], ds_chunked.sizes["y"])

            # Compute chunk geospatial bounds
            lon_start, lat_start = transform * (x_start, y_start)
            lon_end, lat_end = transform * (x_end, y_end)

            # Store chunk bounds
            chunk_bounds[(x, y)] = {
                "lon_start": lon_start,
                "lat_start": lat_start,
                "lon_end": lon_end,
                "lat_end": lat_end,
            }

    return chunk_bounds, img_crs


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
        "date": f"{YEAR}-06-01",
    }


def get_pixels(result):
    pixels = []
    for url, win in result:
        with rasterio.open(url) as src:
            data = src.read(window=win)
            if NODATA in data:
                return
            pixels.append(data)
            # transform = src.window_transform(win)

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

    return pixels


index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 2))

# Setup model components
tfm = v2.Compose([v2.Normalize(mean=MEAN, std=STD)])
ds = ClayDataset(chips_path=[], transform=tfm)

# Load model
rgb_model = CLAYModule.load_from_checkpoint(
    CKPT_PATH,
    mask_ratio=0.0,
    band_groups={"rgb": (2, 1, 0), "nir": (3,)},
    bands=4,
    strict=False,  # ignore the extra parameters in the checkpoint
    embeddings_level="group",
)
# Set the model to evaluation mode
rgb_model.eval()

outdir_embeddings = Path("data/embeddings")
outdir_embeddings.mkdir(exist_ok=True, parents=True)

xoff = index * CHIP_SIZE
yoff = 0
embeddings = []
all_bounds = []
results = []
while yoff < RASTER_Y_SIZE:
    result = tiles_and_windows(Window(xoff, yoff, CHIP_SIZE, CHIP_SIZE))
    if result is not None:
        results.append(result)

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

    print(len(embeddings), len(results))
    embeddings_ = numpy.vstack(embeddings)
    # embeddings_ = embeddings[0]
    print("Embeddings shape: ", embeddings_.shape)

    # remove date and lat/lon
    embeddings_ = embeddings_[:, :-2, :].mean(axis=0)

    print(f"Embeddings have shape {embeddings_.shape}")

    # reshape to disaggregated patches
    embeddings_patch = embeddings_.reshape([2, 16, 16, 768])

    # average over the band groups
    embeddings_mean = embeddings_patch.mean(axis=0)

    print(f"Average patch embeddings have shape {embeddings_mean.shape}")

    if result is not None:
        print("result: ", result[0][0])
        pix = get_pixels(result)
        chunk_bounds, epsg = patch_bounds_from_url(result[0][0])
        # print("chunk_bounds: ", chunk_bounds)
        print("chunk bounds length:", len(chunk_bounds))

        # Iterate through each patch
        for i in range(embeddings_mean.shape[0]):
            for j in range(embeddings_mean.shape[1]):
                embeddings_output_patch = embeddings_mean[i, j]

                item_ = [
                    element
                    for element in list(chunk_bounds.items())
                    if element[0] == (i, j)
                ]
                box_ = [
                    item_[0][1]["lon_start"],
                    item_[0][1]["lat_start"],
                    item_[0][1]["lon_end"],
                    item_[0][1]["lat_end"],
                ]

                data = {
                    "date": pd.to_datetime(batch["date"], format="%Y-%m-%d"),
                    "embeddings": [numpy.ascontiguousarray(embeddings_output_patch)],
                }

                # Define the bounding box as a Polygon (xmin, ymin, xmax, ymax)
                # The box_ list is encoded as
                # [bottom left x, bottom left y, top right x, top right y]
                box_emb = shapely.geometry.box(box_[0], box_[1], box_[2], box_[3])

                print(str(epsg)[-4:])

                # Create the GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    data, geometry=[box_emb], crs=f"EPSG:{str(epsg)[-4:]}"
                )

                # Reproject to WGS84 (lon/lat coordinates)
                gdf = gdf.to_crs(epsg=4326)

                with tempfile.TemporaryDirectory() as tmp:
                    # tmp = "/home/tam/Desktop/wcctmp"

                    outpath = f"{tmp}/worldcover_patch_embeddings_{YEAR}_{index}_{i}_{j}_v{VERSION}.gpq"
                    print(f"Uploading embeddings to {outpath}")
                    # print(gdf)

                    gdf.to_parquet(
                        path=outpath, 
                        compression="ZSTD", 
                        schema_version="1.0.0"
                    )

                    s3_client = boto3.client("s3")
                    s3_client.upload_file(
                        outpath,
                        BUCKET,
                        f"v{VERSION}/{YEAR}/{os.path.basename(outpath)}",
                    )
