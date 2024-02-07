#!/usr/bin/env python3
import os
import tempfile

import boto3
import geopandas as gpd
import numpy
import pyarrow as pa
import rasterio
import torch
from rasterio.windows import Window
from shapely import box

from src.datamodule import ClayDataModule
from src.model_clay import CLAYModule

index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))

YEAR = 2021
DATE = f"{YEAR}-06-01"
RASTER_X_SIZE = (125 - 67) * 12000
RASTER_Y_SIZE = (49 - 24) * 12000
TILE_SIZE = 512
NODATA = 0
CKPT_PATH = "s3://clay-model-ckpt/v0/mae_epoch-24_val-loss-0.46.ckpt"
VERSION = "001"
BUCKET = "clay-worldcover-embeddings"

xoff = index * TILE_SIZE
yoff = 0
all_bounds = []
counter = 0

with tempfile.TemporaryDirectory() as tmp:
    # tmp = "/home/tam/Desktop/wcctmp"
    with rasterio.open(
        f"https://clay-mgrs-samples.s3.amazonaws.com/worldcover_index_usa_{YEAR}.vrt"
    ) as src:
        while yoff < RASTER_Y_SIZE - TILE_SIZE:
            if counter % 100 == 0:
                print("Offset", xoff, yoff)
            counter += 1

            # Read window
            win = Window(xoff, yoff, TILE_SIZE, TILE_SIZE)
            data = src.read(window=win)

            # Check for nodata
            if NODATA in data:
                yoff += TILE_SIZE
                continue

            # Write to local file
            with rasterio.open(
                f"{tmp}/worldcover_{YEAR}_{xoff}_{yoff}.tif",
                "w",
                width=TILE_SIZE,
                height=TILE_SIZE,
                count=4,
                compress="deflate",
                transform=src.window_transform(win),
                nodata=NODATA,
                dtype="float32",
                crs="epsg:4326",
            ) as dst:
                dst.write(data)
                dst.update_tags(date=DATE)

            # Track bounds of tile
            all_bounds.append(src.window_bounds(win))

            yoff += TILE_SIZE

    # Load model
    rgb_model = CLAYModule.load_from_checkpoint(
        CKPT_PATH,
        mask_ratio=0.0,
        band_groups={"rgb": (0, 1, 2), "nir": (3,)},
        bands=4,
        strict=False,  # ignore the extra parameters in the checkpoint
    )
    # Set the model to evaluation mode
    rgb_model.eval()

    # Load the datamodule, with the reduced set of band normalization data
    class ClayDataModuleRGB(ClayDataModule):
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

    dm = ClayDataModuleRGB(data_dir=tmp, batch_size=8)
    dm.setup(stage="predict")
    trn_dl = iter(dm.predict_dataloader())

    # Generate embeddings
    embeddings = []
    for batch in trn_dl:
        with torch.no_grad():
            batch["pixels"] = batch["pixels"].to(rgb_model.device)
            batch["timestep"] = batch["timestep"].to(rgb_model.device)
            batch["latlon"] = batch["latlon"].to(rgb_model.device)
            (
                unmasked_patches,
                unmasked_indices,
                masked_indices,
                masked_matrix,
            ) = rgb_model.model.encoder(batch)

            embeddings.append(unmasked_patches.detach().cpu().numpy())

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

    outpath = f"{tmp}/worldcover_embeddings_{YEAR}_{index}_v{VERSION}.gpq"
    print(f"Uploading embeddings to {outpath}")

    gdf.to_parquet(path=outpath, compression="ZSTD", schema_version="1.0.0")

    s3_client = boto3.client("s3")
    s3_client.upload_file(
        outpath,
        BUCKET,
        f"v{VERSION}/{YEAR}/{os.path.basename(outpath)}",
    )
