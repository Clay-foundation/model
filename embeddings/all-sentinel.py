import gzip
import json
import logging
import os
import sys

import boto3
from pystac import Item
from stacchip.indexer import Sentinel2Indexer

from embeddings.utils import load_clay

logger = logging.getLogger("clay")


# Create 2023 file from full archive

s3 = boto3.resource("s3")


def log(comment):
    sys.stdout.write(f"\r{comment}")


log_every = 100000

count = 0

# with gzip.open(f"data/element84-tiles-2023.gz", "wt") as dst:
#     with gzip.open(f"data/element84-tiles.list.gz") as fl:
#         line = fl.readline()
#         while line:
#             line =  line.decode().rstrip()
#             c = line.split("/")
#             if c[4] == "2019":
#                 line = fl.readline()
#                 continue
#             elif int(c[7]) < 2023:
#                 line = fl.readline()
#                 continue
#             elif not line.endswith("L2A.json"):
#                 line = fl.readline()
#                 continue

#             count += 1
#             if count % log_every == 0:
#                 log(f"Found {count} scenes... {line}")

#             dst.write(line + "\n")
#             line = fl.readline()


def open_manifest(path):
    pass


data = open_manifest()

index = 42

key = data[index].decode().rstrip().replace("s3://sentinel-cogs/", "")

stac_json = json.load(s3.Object("sentinel-cogs", key).get()["Body"])

item = Item.from_dict(stac_json)


indexer = Sentinel2Indexer(item, chip_max_nodata=0.1)


SCENES_LIST = "data/element84-tiles-2023.gz"


def open_scenes_list():
    with gzip.open(SCENES_LIST) as fl:
        return fl.readlines()


def process_scene(clay, path, batchsize):
    pass


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
