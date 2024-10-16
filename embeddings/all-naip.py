import datetime
import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import boto3
from rasterio.errors import RasterioIOError
from rio_stac import create_stac_item
from stacchip.chipper import Chipper
from stacchip.indexer import NoStatsChipIndexer

from embeddings.utils import (
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


MANIFEST = "data/naip-manifest.txt.zip"
EMBEDDINGS_BUCKET = "clay-embeddings-naip"


def open_scene_list():
    """
    Read the naip-analytic manifest file and extract a list of NAIP
    scenes as tif files to process.

    The file used here is the zipped version of the original manifest file.
    """
    with zipfile.ZipFile(MANIFEST) as zf:
        with io.TextIOWrapper(zf.open("naip-manifest.txt"), encoding="utf-8") as f:
            data = f.readlines()
    data = [Path(dat.rstrip()) for dat in data if "rgbir_cog"]
    data = [dat for dat in data if dat.suffix == ".tif"]
    logger.debug(f"Found {len(data)} NAIP scenes in manifest")
    return data


def process_scene(clay, path, batchsize):
    """
    Embeds a slingle NAIP scene.
    """
    state = path.parts[0]
    datestr = path.stem.split("_")[-1]
    date = datetime.datetime(int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8]))
    gsd = float(path.parts[2].replace("cm", "")) / 100
    bands, waves, mean, std = load_metadata("naip")

    logger.debug(f"Processing {path} in state {state} and date {date}")

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".tif") as fl:
        s3 = boto3.client("s3")
        s3.download_fileobj(
            "naip-analytic", str(path), fl, ExtraArgs={"RequestPayer": "requester"}
        )

        item = create_stac_item(fl.name, with_proj=True)

        item.datetime = date
        item.id = f"{state}_{path.stem}"

        try:
            indexer = NoStatsChipIndexer(item)
            chipper = Chipper(indexer)
            bboxs, datetimes, pixels = get_pixels(
                item=item, indexer=indexer, chipper=chipper
            )
        except RasterioIOError:
            logger.warning("Skipping scene due to rasterio io error")
            return

        time_norm, latlon_norm, gsd, pixels_norm = prepare_datacube(
            mean=mean, std=std, datetimes=datetimes, bboxs=bboxs, pixels=pixels, gsd=gsd
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
        # Write class embeddings
        kwargs = dict(
            bboxs=bboxs,
            datestr=datestr,
            gsd=gsd,
            destination_bucket=EMBEDDINGS_BUCKET,
            path=path,
        )
        logger.debug("Writing class embeddings")
        write_to_table(embeddings=cls_embeddings, **kwargs)


def process():
    if "AWS_BATCH_JOB_ARRAY_INDEX" not in os.environ:
        raise ValueError("AWS_BATCH_JOB_ARRAY_INDEX env var not set")
    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))
    items_per_job = int(os.environ.get("ITEMS_PER_JOB", 2))
    batchsize = int(os.environ.get("EMBEDDING_BATCH_SIZE", 50))

    scenes = open_scene_list()
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
