import gzip
import json
import logging
import os

import boto3
from pystac import Item
from rasterio.errors import RasterioIOError
from stacchip.chipper import Chipper
from stacchip.indexer import Sentinel2Indexer

from embeddings.utils import (
    get_embeddings,
    get_pixels,
    load_clay,
    load_metadata,
    prepare_datacube,
    write_to_table,
)

logger = logging.getLogger("clay")


SCENES_LIST = "data/element84-tiles-2023.gz"
EMBEDDINGS_BUCKET = "clay-embeddings-sentinel-2"
GSD = 10


def open_scenes_list():
    with gzip.open(SCENES_LIST) as fl:
        data = fl.readlines()
    return [dat.decode().rstrip() for dat in data]


def process_scene(clay, path, batchsize):
    bands, waves, mean, std = load_metadata("sentinel_2_l2a")

    key = path.replace("s3://sentinel-cogs/", "")

    s3 = boto3.resource("s3")

    stac_json = json.load(s3.Object("sentinel-cogs", key).get()["Body"])

    item = Item.from_dict(stac_json)

    bands, waves, mean, std = load_metadata("naip")

    try:
        indexer = Sentinel2Indexer(item, chip_max_nodata=0.1)
        chipper = Chipper(item, assets=bands)
        bboxs, datetimes, pixels = get_pixels(
            item=item, indexer=indexer, chipper=chipper
        )
    except RasterioIOError:
        logger.warning("Skipping scene due to rasterio io error")
        return

    time_norm, latlon_norm, gsd, pixels_norm = prepare_datacube(
        mean=mean, std=std, datetimes=datetimes, bboxs=bboxs, pixels=pixels, gsd=GSD
    )
    # Embed data
    cls_embeddings, patch_embeddings = get_embeddings(
        clay=clay,
        pixels_norm=pixels_norm,
        time_norm=time_norm,
        latlon_norm=latlon_norm,
        waves=waves,
        gsd=gsd,
        batchsize=batchsize,
    )
    kwargs = dict(
        bboxs=bboxs,
        datestr=str(item.datetime.date()),
        gsd=gsd,
        destination_bucket=EMBEDDINGS_BUCKET,
        path=path,
    )

    write_to_table(embeddings=cls_embeddings, **kwargs)
    write_to_table(embeddings=patch_embeddings, **kwargs)


def process():
    if "AWS_BATCH_JOB_ARRAY_INDEX" not in os.environ:
        raise ValueError("AWS_BATCH_JOB_ARRAY_INDEX env var not set")
    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))
    items_per_job = int(os.environ.get("ITEMS_PER_JOB", 100))
    batchsize = int(os.environ.get("EMBEDDING_BATCH_SIZE", 50))

    scenes = open_scenes_list()
    clay = load_clay()

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
