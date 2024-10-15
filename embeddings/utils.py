import logging
import math

import boto3
import geoarrow.pyarrow as ga
import numpy as np
import pyarrow as pa
import torch
import yaml
from box import Box
from geoarrow.pyarrow import io as gaio
from torchvision.transforms import v2

from src.module import ClayMAEModule

CHECKPOINT = "data/clay-model-v1.5.0-october-12.ckpt"
EMBEDDING_SHAPE_CLASS = 2
EMBEDDING_SHAPE_PATCH = 3

logger = logging.getLogger("clay")


def load_metadata(platform):
    metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))
    platform_meta = getattr(metadata, platform)

    bands = list(platform_meta.bands.wavelength.keys())
    waves = list(platform_meta.bands.wavelength.values())
    mean = list(platform_meta.bands.mean.values())
    std = list(platform_meta.bands.std.values())

    return bands, waves, mean, std


def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def prepare_datacube(mean, std, datetimes, bboxs, pixels, gsd):
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

    return time_norm, latlon_norm, gsd, pixels_norm


def get_pixels(item, indexer, chipper):
    chips = []
    datetimes = []
    bboxs = []
    chip_ids = []
    item_ids = []
    for x, y, chip in chipper:
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
    cls_embeddings = None
    for i in range(0, len(pixels_norm), batchsize):
        if i / batchsize % 5 == 0:
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
            unmsk_patch, unmsk_idx, msk_idx, msk_matrix = clay.model.encoder(datacube)
        # The first embedding is the class token, which is the
        # overall single embedding we want to keep.
        if cls_embeddings is None:
            cls_embeddings = unmsk_patch[:, 0, :]
            patch_embeddings = unmsk_patch[:, 1:, :]
        else:
            cls_embeddings = torch.vstack((cls_embeddings, unmsk_patch[:, 0, :]))
            patch_embeddings = torch.vstack((patch_embeddings, unmsk_patch[:, 1:, :]))

    return cls_embeddings, patch_embeddings


def load_clay():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=CHECKPOINT,
        metadata_path="configs/metadata.yaml",
        model_size="large",
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0,
        shuffle=False,
    )
    model.eval()

    return model.to(device)


def write_to_table(embeddings, bboxs, datestr, gsd, destination_bucket, path):
    np_embeddings = embeddings.cpu().numpy()
    index = {"geometry": ga.as_geoarrow([dat.wkt for dat in bboxs])}
    if len(embeddings.shape) == EMBEDDING_SHAPE_CLASS:
        # Handle class embeddings
        index["embeddings"] = [np.ascontiguousarray(dat) for dat in np_embeddings]
        embedding_level = "class"
    elif len(embeddings.shape) == EMBEDDING_SHAPE_PATCH:
        # Handle patch embeddings
        for i in range(embeddings.shape[1]):
            index[f"patch_embeddings_{i}"] = [
                np.ascontiguousarray(dat) for dat in np_embeddings[:, i, :]
            ]
        embedding_level = "patch"

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
        Key=f"{embedding_level}/{path.parent}/{path.stem}.parquet",
    )
