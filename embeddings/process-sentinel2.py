#!/usr/bin/env python3
import gc
import gzip
import json
import logging
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import boto3
import numpy as np
import rasterio
import torch
from pystac import Item
from stacchip.chipper import Chipper
from stacchip.indexer import Sentinel2Indexer

from embeddings.utils import (
    check_exists,
    get_embeddings,
    get_pixels,
    load_clay,
    load_metadata,
    prepare_datacube,
    write_to_table,
)

logging.basicConfig()
logger = logging.getLogger("clay")
logger.setLevel(logging.DEBUG)

SCENES_LIST = "data/stac_manifest_brazil_sentinel_2023_2024.csv.gz"
EMBEDDINGS_BUCKET = os.environ["EMBEDDINGS_BUCKET"]


GSD = 10
S2_BUCKET = "sentinel-2-cogs"

# Increase thresholds significantly to see if we can get any valid pixels
# at the scene level.
CLOUD_LIMIT = 0.9  # Allow up to 80% cloud coverage
NODATA_LIMIT = 0.5  # Allow up to 30% nodata


def open_scenes_list():
    with gzip.open(SCENES_LIST) as fl:
        data = fl.readlines()
    data = [dat.decode().rstrip() for dat in data]
    data = [dat for dat in data if dat.split("/")[7] == "2024"]
    # Process the X, C, and D regions last
    data = sorted(data, key=lambda dat: dat.split("/")[5] in ["X", "C", "D"])
    data = [Path(dat.replace("s3://sentinel-cogs/", "")) for dat in data]
    logger.debug(f"Found {len(data)} scenes to process")
    return data


def download_scenes_local(tmp, item, bands):
    s3 = boto3.client("s3")
    for band in bands:
        local_asset_path = f"{tmp}/{band}.tif"
        remote_asset_key = item.assets[band].href.replace(
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/", ""
        )
        print(f"Downloading band {band} to {local_asset_path}")
        with open(local_asset_path, mode="w+b") as fl:
            s3.download_fileobj("sentinel-cogs", remote_asset_key, fl)
        item.assets[band].href = local_asset_path

    return item


def check_scene_quality(item, nodata_limit=0.3, cloud_limit=0.8):
    """Check if scene has acceptable cloud/nodata coverage
    before downloading all bands."""
    logger.debug("Checking scene quality via SCL band")

    with rasterio.open(item.assets["scl"].href) as src:
        scl = src.read(1)
        total_pixels = scl.size

        # Calculate scene-level statistics
        unique_values, counts = np.unique(scl, return_counts=True)
        stats = {val: count / total_pixels for val, count in zip(unique_values, counts)}

        # Log SCL composition
        logger.info("\nSCL Analysis:")
        for val, percentage in stats.items():
            logger.info(f"SCL value {val}: {percentage*100:.2f}%")

        # Check against Sentinel2Indexer's filter values
        cloud_percentage = sum(stats.get(val, 0) for val in Sentinel2Indexer.scl_filter)
        nodata_percentage = stats.get(0, 0)  # 0 is NO_DATA in SCL

        logger.info(
            f"\nScene-level statistics:"
            f"\nCloud coverage: {cloud_percentage*100:.1f}%"
            f"\nNodata coverage: {nodata_percentage*100:.1f}%"
            f"\nLimits - Cloud: {cloud_limit*100}%, Nodata: {nodata_limit*100}%"
        )

        if cloud_percentage > cloud_limit:
            logger.warning(
                f"Scene rejected: {cloud_percentage*100:.1f}% cloud coverage "
                f"exceeds {cloud_limit*100}% limit"
            )
            return False

        if nodata_percentage > nodata_limit:
            logger.warning(
                f"Scene rejected: {nodata_percentage*100:.1f}% nodata "
                f"exceeds {nodata_limit*100}% limit"
            )
            return False

        return True


def process_scene(clay, path, batchsize):  # noqa: PLR0915
    """Process a single scene, checking quality before downloading all bands."""
    try:
        scene_start_time = time.time()
        logger.debug(f"Starting to process scene: {path}")

        bands, waves, mean, std = load_metadata("sentinel-2-l2a")
        logger.debug(f"Loaded metadata with bands: {bands}")

        s3 = boto3.resource("s3")
        stac_json = json.load(s3.Object("sentinel-cogs", str(path)).get()["Body"])
        item = Item.from_dict(stac_json)

        # Check scene quality before downloading other bands
        if not check_scene_quality(item, NODATA_LIMIT, CLOUD_LIMIT):
            return

        # If scene passes quality check, proceed with full processing
        bands, waves, mean, std = load_metadata("sentinel-2-l2a")
        logger.debug(f"Scene passed quality check. Loading bands: {bands}")

        with tempfile.TemporaryDirectory() as tmp:
            download_start = time.time()
            item = download_scenes_local(tmp, item, bands)
            download_time = time.time() - download_start
            logger.info(f"Download time: {timedelta(seconds=download_time)}")

            indexer = Sentinel2Indexer(item, chip_max_nodata=NODATA_LIMIT)
            chipper = Chipper(indexer, assets=bands)
            logger.debug(f"Creating chips for {item.id}")
            STEP = 50

            all_bboxs = []
            all_cls_embeddings = None
            total_chips = 0

            for index in range(0, len(chipper), STEP):
                try:
                    bboxs, datetimes, pixels = get_pixels(
                        item=item,
                        indexer=indexer,
                        chipper=chipper,
                        start=index,
                        end=index + STEP,
                    )

                    if len(pixels) == 0:
                        logger.warning(f"No valid chips in batch {index}-{index+STEP}")
                        continue

                    logger.info(f"Doing batch of {len(pixels)} chips")

                    time_norm, latlon_norm, gsd, pixels_norm = prepare_datacube(
                        mean=mean,
                        std=std,
                        datetimes=datetimes,
                        bboxs=bboxs,
                        pixels=pixels,
                        gsd=GSD,
                    )

                    cls_embeddings = get_embeddings(
                        clay=clay,
                        pixels_norm=pixels_norm,
                        time_norm=time_norm,
                        latlon_norm=latlon_norm,
                        waves=waves,
                        gsd=gsd,
                        batchsize=batchsize,
                    )
                    logger.info(
                        f"Created embeddings with shape: {cls_embeddings.shape}"
                    )

                    all_bboxs += bboxs
                    if all_cls_embeddings is None:
                        all_cls_embeddings = cls_embeddings
                    else:
                        all_cls_embeddings = np.vstack(
                            (all_cls_embeddings, cls_embeddings)
                        )
                    total_chips += len(pixels)
                    logger.info(f"Total embeddings so far: {all_cls_embeddings.shape}")

                    # Explicitly free memory after each batch
                    del pixels, time_norm, latlon_norm, gsd, pixels_norm
                    gc.collect()
                    torch.cuda.empty_cache()  # Free GPU memory if using CUDA

                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU OUT OF MEMORY during chip processing: {e}")
                    # Try to free memory and continue with next batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                except Exception as e:
                    logger.error(f"Error processing batch {index}-{index+STEP}: {e}")
                    continue

            if not all_bboxs or all_cls_embeddings is None:
                logger.warning(f"No embeddings generated for scene {path}")
                return

            logger.info(
                f"Writing {len(all_bboxs)} embeddings to "
                + f'"{EMBEDDINGS_BUCKET}/{path.parent}/{path.stem}.parquet"'
            )

            kwargs = dict(
                bboxs=all_bboxs,
                datestr=str(item.datetime.date()),
                gsd=GSD,
                destination_bucket=EMBEDDINGS_BUCKET,
                path=path,
                source_bucket="sentinel-cogs",
            )

            write_to_table(embeddings=all_cls_embeddings, **kwargs)
            logger.info(f"Successfully wrote embeddings for scene {path}")

            total_time = time.time() - scene_start_time
            logger.info(
                f"Scene processing completed:"
                f"\n  Total time: {timedelta(seconds=total_time)}"
                f"\n  Total chips processed: {total_chips}"
                f"\n  Final embeddings shape: {all_cls_embeddings.shape}"
            )

        # Clean up at the end
        del all_cls_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Failed to process scene {path}: {e}")
        # Ensure memory is freed even if we hit an error
        gc.collect()
        torch.cuda.empty_cache()


def process():
    if "AWS_BATCH_JOB_ARRAY_INDEX" not in os.environ:
        raise ValueError("AWS_BATCH_JOB_ARRAY_INDEX env var not set")

    # Add device logging
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))
    items_per_job = int(os.environ.get("ITEMS_PER_JOB", 100))
    batchsize = int(os.environ.get("EMBEDDING_BATCH_SIZE", 50))

    scenes = open_scenes_list()
    clay = load_clay()

    # Calculate start and end indices
    start_idx = index * items_per_job
    end_idx = (index + 1) * items_per_job
    
    # Check if start index is beyond array bounds
    if start_idx >= len(scenes):
        logger.warning(f"Job index {index} exceeds number of available scenes ({len(scenes)}). Nothing to process.")
        return

    # Adjust end index if it exceeds array bounds
    end_idx = min(end_idx, len(scenes))
    
    logger.info(f"Processing scenes {start_idx} to {end_idx-1} out of {len(scenes)} total scenes")

    for i in range(start_idx, end_idx):
        if check_exists(scenes[i]):
            logger.debug(f"Skipping scene because exists: {scenes[i]}")
            continue

        process_scene(
            clay=clay,
            path=scenes[i],
            batchsize=batchsize,
        )


if __name__ == "__main__":
    logger.debug("Starting")
    process()
    logger.debug("Done!")
