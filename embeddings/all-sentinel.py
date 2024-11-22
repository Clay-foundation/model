import gzip
import json
import logging
import os
import tempfile
from pathlib import Path

import boto3
import numpy as np
from pystac import Item
from stacchip.chipper import Chipper
from stacchip.indexer import Sentinel2Indexer

from embeddings.utils import (
    get_embeddings,
    get_pixels,
    load_metadata,
    prepare_datacube,
    write_to_table,
)

logging.basicConfig()
logger = logging.getLogger("clay")
logger.setLevel(logging.DEBUG)

SCENES_LIST = "data/element84-tiles-2023.gz"
EMBEDDINGS_BUCKET = "clay-embeddings-sentinel-2"
GSD = 10
S2_BUCKET = "sentinel-2-cogs"


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


def process_scene(clay, path, batchsize):
    bands, waves, mean, std = load_metadata("sentinel-2-l2a")

    s3 = boto3.resource("s3")

    stac_json = json.load(s3.Object("sentinel-cogs", str(path)).get()["Body"])

    item = Item.from_dict(stac_json)

    # Sanity checks
    if "red" not in item.assets:
        logger.debug(f"No red band for {path}")
        return
    elif not item.ext.has("proj"):
        logger.debug(f"No proj for {path}")
        return

    all_bboxs = []
    all_cls_embeddings = None

    with tempfile.TemporaryDirectory() as tmp:
        item = download_scenes_local(tmp, item, bands)
        indexer = Sentinel2Indexer(item, chip_max_nodata=0.1)
        chipper = Chipper(indexer, assets=bands)
        logger.debug(f"Creating chips for {item.id}")
        STEP = 50
        for index in range(0, len(chipper), STEP):
            bboxs, datetimes, pixels = get_pixels(
                item=item,
                indexer=indexer,
                chipper=chipper,
                start=index,
                end=index + STEP,
            )

            if not len(pixels):
                continue

            time_norm, latlon_norm, gsd, pixels_norm = prepare_datacube(
                mean=mean,
                std=std,
                datetimes=datetimes,
                bboxs=bboxs,
                pixels=pixels,
                gsd=GSD,
            )

            # Embed data
            cls_embeddings = get_embeddings(
                clay=clay,
                pixels_norm=pixels_norm,
                time_norm=time_norm,
                latlon_norm=latlon_norm,
                waves=waves,
                gsd=gsd,
                batchsize=batchsize,
            )
            all_bboxs += bboxs
            if all_cls_embeddings is None:
                all_cls_embeddings = cls_embeddings
            else:
                all_cls_embeddings = np.vstack((all_cls_embeddings, cls_embeddings))

    kwargs = dict(
        bboxs=all_bboxs,
        datestr=str(item.datetime.date()),
        gsd=gsd,
        destination_bucket=EMBEDDINGS_BUCKET,
        path=path,
        source_bucket="sentinel-cogs",
    )

    write_to_table(embeddings=all_cls_embeddings, **kwargs)


def process():
    if "AWS_BATCH_JOB_ARRAY_INDEX" not in os.environ:
        raise ValueError("AWS_BATCH_JOB_ARRAY_INDEX env var not set")
    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))
    items_per_job = int(os.environ.get("ITEMS_PER_JOB", 100))
    batchsize = int(os.environ.get("EMBEDDING_BATCH_SIZE", 50))

    scenes = open_scenes_list()
    # clay = load_clay()
    clay = None

    for i in range(index * items_per_job, (index + 1) * items_per_job):
        process_scene(
            clay=clay,
            path=scenes[i],
            batchsize=batchsize,
        )


if __name__ == "__main__":
    logger.debug("Starting")
    process()
    logger.debug("Done!")
