import datetime
import io
import logging
import math
import os
import tempfile
import zipfile

import boto3
import geoarrow.pyarrow as ga
import numpy as np
import pyarrow as pa
import torch
from geoarrow.pyarrow import io as gaio
from rasterio.errors import RasterioIOError
from rio_stac import create_stac_item
from stacchip.chipper import Chipper
from stacchip.indexer import NoStatsChipIndexer
from torchvision.transforms import v2

from src.module import ClayMAEModule

logging.basicConfig()
logger = logging.getLogger("clay")
logger.setLevel(logging.DEBUG)


def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def prepare_datacube(datetimes, bboxs, pixels, gsd):
    # Set mean, std, and wavelengths metadata
    mean = [
        110.16,
        115.41,
        98.15,
        139.04,
    ]
    std = [47.23, 39.82, 35.43, 49.86]
    waves = [0.65, 0.56, 0.48, 0.842]

    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )

    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]
    time_norm = np.hstack((week_norm, hour_norm))

    latlons = [normalize_latlon(*bbox.centroid.coords[0]) for bbox in bboxs]
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]
    latlon_norm = np.hstack((lat_norm, lon_norm))

    gsd = [gsd]

    pixels_norm = transform(pixels)

    return waves, time_norm, latlon_norm, gsd, pixels_norm


def get_pixels(item):
    indexer = NoStatsChipIndexer(item)

    # Instanciate the chipper
    chipper = Chipper(indexer)

    # Get first chip for the "image" asset key
    chips = []
    datetimes = []
    bboxs = []
    chip_ids = []
    item_ids = []
    for idx, (x, y, chip) in enumerate(chipper):
        chips.append(chip)
        datetimes.append(item.datetime)
        bboxs.append(indexer.get_chip_bbox(x, y))
        chip_ids.append((x, y))
        item_ids.append(item.id)

    pixels = np.array([np.array(list(chip.values())).squeeze() for chip in chips])
    return bboxs, datetimes, pixels


def get_embeddings(clay, pixels_norm, time_norm, latlon_norm, waves, gsd, batchsize):  # noqa: PLR0913
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug(f"Using device {device} to create {len(pixels_norm)} embeddings")
    # Run the clay encoder
    embeddings = None
    for i in range(0, len(pixels_norm), batchsize):
        if i % 500 == 0:
            logger.debug(f"Iteration {i}")
        datacube = {
            "pixels": torch.tensor(
                pixels_norm[i : (i + batchsize)], dtype=torch.float32, device=device
            ),
            "time": torch.tensor(
                time_norm[i : (i + batchsize)], dtype=torch.float32, device=device
            ),
            "latlon": torch.tensor(
                latlon_norm[i : (i + batchsize)], dtype=torch.float32, device=device
            ),
            "waves": torch.tensor(waves, dtype=torch.float32, device=device),
            "gsd": torch.tensor(gsd, dtype=torch.float32, device=device),
            "platform": ["naip"],
        }
        with torch.no_grad():
            cls_embedding = clay(datacube)
        if embeddings is None:
            embeddings = cls_embedding
        else:
            embeddings = torch.vstack((embeddings, cls_embedding))

    return embeddings


def open_scene_list():
    with zipfile.ZipFile("/data/naip-manifest.txt.zip") as zf:
        with io.TextIOWrapper(zf.open("naip-manifest.txt"), encoding="utf-8") as f:
            data = f.readlines()
    data = [dat.rstrip() for dat in data if "rgbir_cog" in dat]
    logger.debug(f"Found {len(data)} NAIP scenes in manifest")
    return data


def load_clay():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClayMAEModule.load_from_checkpoint(
        checkpoint_path="/data/clay-model-v1.5.0-september-30.ckpt",
        metadata_path="configs/metadata.yaml",
        model_size="large",
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0,
        shuffle=False,
    )
    model.eval()

    return model.to(device)


def write_to_table(embeddings, bboxs, datestr, gsd, destination_bucket, path, item_id):  # noqa: PLR0913
    index = {
        "embeddings": [np.ascontiguousarray(dat) for dat in embeddings.cpu().numpy()],
        "geometry": ga.as_geoarrow([dat.wkt for dat in bboxs]),
    }

    table = pa.table(
        index,
        metadata={
            "date": datestr,
            "gsd": str(gsd[0]),
            "uri": f"s3://naip-analytic/{path}",
        },
    )

    writer = pa.BufferOutputStream()
    gaio.write_geoparquet_table(table, writer)
    body = bytes(writer.getvalue())
    s3_resource = boto3.resource("s3")
    s3_bucket = s3_resource.Bucket(name=destination_bucket)
    s3_bucket.put_object(
        Body=body,
        Key=f"{item_id}.parquet",
    )


def process_scene(clay, path, destination_bucket, batchsize):
    state = path.split("/")[0]
    datestr = path.split("/")[-1].split("_")[-1].split(".txt")[0]
    gsd = float(path.split("/")[2].replace("cm", "")) / 100
    date = datetime.datetime(int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8]))
    logger.debug(f"Processing {path} in state {state} and date {date}")

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".tif") as f:
        s3 = boto3.client("s3")
        s3.download_fileobj(
            "naip-analytic", path, f, ExtraArgs={"RequestPayer": "requester"}
        )

        item = create_stac_item(f.name, with_proj=True)
        item.datetime = date
        item.id = f"{state}_{path.split('/')[-1].replace('.tif', '')}"

        try:
            bboxs, datetimes, pixels = get_pixels(item)
        except RasterioIOError:
            logger.debug("Skipping scene due to rasterio io error")
            return

        waves, time_norm, latlon_norm, gsd, pixels_norm = prepare_datacube(
            datetimes=datetimes, bboxs=bboxs, pixels=pixels, gsd=gsd
        )

        embeddings = get_embeddings(
            clay=clay,
            pixels_norm=pixels_norm,
            time_norm=time_norm,
            latlon_norm=latlon_norm,
            waves=waves,
            gsd=gsd,
            batchsize=batchsize,
        )

        write_to_table(
            embeddings=embeddings,
            bboxs=bboxs,
            datestr=datestr,
            gsd=gsd,
            destination_bucket=destination_bucket,
            path=path,
            item_id=item.id,
        )


def process():
    if "AWS_BATCH_JOB_ARRAY_INDEX" not in os.environ:
        raise ValueError("AWS_BATCH_JOB_ARRAY_INDEX env var not set")
    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))
    items_per_job = int(os.environ.get("ITEMS_PER_JOB", 100))
    batchsize = int(os.environ.get("EMBEDDING_BATCH_SIZE", 50))
    destination_bucket = "clay-v1-naip-embeddings"

    scenes = open_scene_list()
    clay = load_clay()

    for i in range(index * items_per_job, (index + 1) * items_per_job):
        process_scene(
            clay=clay,
            path=scenes[i],
            destination_bucket=destination_bucket,
            batchsize=batchsize,
        )


if __name__ == "__main__":
    logger.debug("Starting")
    process()
    logger.debug("Done!")
