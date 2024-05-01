#!/usr/bin/env python3

import sys
sys.path.append("../../")

import os
import tempfile
from math import floor
from pathlib import Path
import requests

import boto3
import einops
import geopandas as gpd
import pandas as pd
import numpy
import pyarrow as pa
import rasterio
import shapely
import torch
import xarray as xr
from rasterio.windows import Window
from shapely import box
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

RASTER_X_SIZE = (E_W_INDEX_END - E_W_INDEX_START) * TILE_SIZE
RASTER_Y_SIZE = (N_S_INDEX_END - N_S_INDEX_START) * TILE_SIZE
NODATA = 0
CKPT_PATH = (
    "https://huggingface.co/made-with-clay/Clay/resolve/main/"
    "Clay_v0.1_epoch-24_val-loss-0.46.ckpt"
)
VERSION = "006"
BUCKET = "clay-worldcover-embeddings"
URL_RGBNIR = "https://esa-worldcover-s2.s3.amazonaws.com/rgbnir/{year}/N{yidx}/ESA_WorldCover_10m_{year}_v{version}_N{yidx}W{xidx}_S2RGBNIR.tif"
URL_SWIR = "https://esa-worldcover-s2.s3.amazonaws.com/swir/{year}/N{yidx}/ESA_WorldCover_10m_{year}_v{version}_N{yidx}W{xidx}_SWIR.tif"
WC_VERSION_LOOKUP = {
    2020: 100,
    2021: 200,
}

# Mean and standard deviation for RGBNIR bands
MEAN_RGBNIR = [1369.03, 1597.68, 1741.10, 2858.43]
STD_RGBNIR = [2026.96, 2011.88, 2146.35, 2016.38]

# Mean and standard deviation for SWIR bands
MEAN_SWIR = [2303.00, 1807.79]
STD_SWIR = [1679.88, 1568.06]

grid = gpd.read_file(
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
            URL_RGBNIR.format(
                yidx=y_tile_index,
                xidx=str(x_tile_index).zfill(3),
                year=YEAR,
                version=WC_VERSION_LOOKUP[YEAR],
            ),
            URL_SWIR.format(
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
                URL_RGBNIR.format(
                    yidx=y_tile_index,
                    xidx=str(x_tile_index - 1).zfill(3),
                    year=YEAR,
                    version=WC_VERSION_LOOKUP[YEAR],
                ),
                URL_SWIR.format(
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
                URL_RGBNIR.format(
                    yidx=y_tile_index - 1,
                    xidx=str(x_tile_index).zfill(3),
                    year=YEAR,
                    version=WC_VERSION_LOOKUP[YEAR],
                ),
                URL_SWIR.format(
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
                URL_RGBNIR.format(
                    yidx=y_tile_index - 1,
                    xidx=str(x_tile_index - 1).zfill(3),
                    year=YEAR,
                    version=WC_VERSION_LOOKUP[YEAR],
                ),
                URL_SWIR.format(
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
    # Download the image from the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        return response.content  # Return the image content
    else:
        raise Exception("Failed to download the image")

def patch_bounds_from_url(url, chunk_size=(PATCH_SIZE, PATCH_SIZE)):
    # Download the image from the URL
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
    rgb_bands = []
    swir_bands = []

    for url_rgb, url_swir, win in result:
        with rasterio.open(url_rgb) as src_rgb, rasterio.open(url_swir) as src_swir:
            data_rgb = src_rgb.read(window=win)
            data_swir = src_swir.read(window=win)
            if NODATA in data_rgb or NODATA in data_swir:
                return
            transform = src_rgb.window_transform(win)
            rgb_bands.append(data_rgb)
            swir_bands.append(data_swir)

    if len(rgb_bands) == 0 or len(swir_bands) == 0:
        return

    rgb_data = numpy.vstack(rgb_bands)
    #rgb_data = rgb_data.transpose(1,2,0)
    swir_data = numpy.vstack(swir_bands)
    #swir_data = swir_data.transpose(1,2,0)
    print("rgb_data: ", rgb_data.shape)
    print("swir_data: ", swir_data.shape)

    if rgb_data.shape[0] == 1:
        rgb_data = rgb_data[0]
    elif rgb_data.shape[0] == 2:
        if rgb_data.shape[2] == CHIP_SIZE:
            rgb_data = einops.pack(rgb_data, "b * w")[0]
            swir_data = einops.pack(swir_data, "b * w")[0]
            print("swir_data r1: ", swir_data.shape)
        else:
            rgb_data = einops.pack(rgb_data, "b h *")[0]
            swir_data = einops.pack(swir_data, "b h *")[0]
            print("swir_data r2: ", swir_data.shape)
    else:
        rgb_px1 = einops.pack(rgb_data[:2], "b w *")[0]
        rgb_px2 = einops.pack(rgb_data[2:], "b w *")[0]
        #rgb_data = einops.pack((rgb_px1, rgb_px2), "b * w")[0]
        print("rgb_data re: ", rgb_data.shape)

        #swir_px1 = einops.pack(swir_data[:2], "b w *")[0]
        #print("swir_data re: ", swir_px1.shape)
        #swir_px2 = einops.pack(swir_data[2:], "b w *")[0]
        #swir_data = einops.pack((swir_px1, swir_px2), "b * w")[0]
        print("swir_data re: ", swir_data.shape)

    rgb_data = rgb_data.transpose(1,2,0)
    swir_data = swir_data.transpose(1,2,0)
    print("rgb_data: ", rgb_data.shape)
    print("swir_data: ", swir_data.shape)
    combined_data = numpy.concatenate((rgb_data,swir_data), axis=-1) #numpy.concatenate([rgb_data, swir_data])
    combined_data = combined_data.transpose(2,0,1)
    print("combined_data: ", combined_data.shape)

    return {
        "pixels": torch.as_tensor(data=[combined_data], dtype=torch.float32).to(rgb_model.device),
        "latlon": torch.as_tensor(data=[ds.normalize_latlon(transform[0], transform[3])]).to(rgb_model.device),
        "timestep": torch.as_tensor(data=[ds.normalize_timestamp(f"{YEAR}-06-01")]).to(rgb_model.device),
        "date": f"{YEAR}-06-01"
    }



index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 2))

# Setup model components
tfm = v2.Compose([v2.Normalize(mean=MEAN_RGBNIR + MEAN_SWIR, std=STD_RGBNIR + STD_SWIR)])
ds = ClayDataset(chips_path=[], transform=tfm)

# Load model
rgb_model = CLAYModule.load_from_checkpoint(
    CKPT_PATH,
    mask_ratio=0.0,
    band_groups={"rgb": (2, 1, 0), "nir": (3,), "swir": (4, 5)},
    bands=6,
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
    #embeddings_ = embeddings[0]
    print("Embeddings shape: ", embeddings_.shape)

    # remove date and lat/lon
    embeddings_ = embeddings_[:, :-2, :].mean(axis=0)

    print(f"Embeddings have shape {embeddings_.shape}")

    # reshape to disaggregated patches
    embeddings_patch = embeddings_.reshape([3, 16, 16, 768])

    # average over the band groups
    embeddings_mean = embeddings_patch.mean(axis=0)

    print(f"Average patch embeddings have shape {embeddings_mean.shape}")

    if result is not None:
        print("result: ", result[0][0])
        #pix = get_pixels(result)
        chunk_bounds, epsg = patch_bounds_from_url(result[0][0])
        #print("chunk_bounds: ", chunk_bounds)
        print("chunk bounds length:", len(chunk_bounds))


        # Iterate through each patch
        for i in range(embeddings_mean.shape[0]):
            for j in range(embeddings_mean.shape[1]):
                embeddings_output_patch = embeddings_mean[i, j]
        
                item_ = [
                    element for element in list(chunk_bounds.items()) if element[0] == (i, j)
                ]
                box_ = [
                    item_[0][1]["lon_start"],
                    item_[0][1]["lat_start"],
                    item_[0][1]["lon_end"],
                    item_[0][1]["lat_end"],
                ]

                data = {
                    #"source_url": batch["source_url"][0],
                    #"date": pd.to_datetime(arg=date, format="%Y-%m-%d").astype(
                    #    dtype="date32[day][pyarrow]"
                    #),
                    #"date": pd.to_datetime(date, format="%Y-%m-%d", dtype="date32[day][pyarrow]"),
                    "date": pd.to_datetime(batch["date"], format="%Y-%m-%d"),
                    "embeddings": [numpy.ascontiguousarray(embeddings_output_patch)],
                }
        
                # Define the bounding box as a Polygon (xmin, ymin, xmax, ymax)
                # The box_ list is encoded as
                # [bottom left x, bottom left y, top right x, top right y]
                box_emb = shapely.geometry.box(box_[0], box_[1], box_[2], box_[3])

                print(str(epsg)[-4:])
        
                # Create the GeoDataFrame
                gdf = gpd.GeoDataFrame(data, geometry=[box_emb], crs=f"EPSG:{str(epsg)[-4:]}")
        
                # Reproject to WGS84 (lon/lat coordinates)
                gdf = gdf.to_crs(epsg=4326)
        
                with tempfile.TemporaryDirectory() as tmp:
                    # tmp = "/home/tam/Desktop/wcctmp"
                
                    outpath = f"{tmp}/worldcover_patch_embeddings_{YEAR}_{index}_{i}_{j}_v{VERSION}.gpq"
                    print(f"Uploading embeddings to {outpath}")
                    #print(gdf)
                
                    gdf.to_parquet(path=outpath, compression="ZSTD", schema_version="1.0.0")
                
                    s3_client = boto3.client("s3")
                    s3_client.upload_file(
                        outpath,
                        BUCKET,
                        f"v{VERSION}/{YEAR}/{os.path.basename(outpath)}",
                    )

