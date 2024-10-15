import gzip
import json
import logging
import os

import boto3
from pystac import Item
from stacchip.indexer import Sentinel2Indexer

from embeddings.utils import load_clay

logger = logging.getLogger("clay")


SCENES_LIST = "data/element84-tiles-2023.gz"


def open_scenes_list():
    with gzip.open(SCENES_LIST) as fl:
        data = fl.readlines()
    return [dat.decode().rstrip() for dat in data]


def process_scene(clay, path, batchsize):
    key = path.replace("s3://sentinel-cogs/", "")

    s3 = boto3.resource("s3")

    stac_json = json.load(s3.Object("sentinel-cogs", key).get()["Body"])

    item = Item.from_dict(stac_json)

    indexer = Sentinel2Indexer(item, chip_max_nodata=0.1)

    return indexer


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
