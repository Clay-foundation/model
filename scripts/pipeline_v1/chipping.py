import argparse
import json
import math
from pathlib import Path

import rasterio
from indexing import get_shape
from pystac import Item
from rasterio.windows import Window

CHIP_SIZE = 256
CHIP_DTYPE = "float32"

LANDSAT_ASSETS = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]
SENTINEL_ASSETS = [
    "aot",
    "blue",
    "coastal",
    "green",
    "nir",
    "nir08",
    "nir09",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "swir22",
]
NAIP_ASSETS = ["asset"]
NZ_ASSETS = ["asset"]

ASSET_LOOKUP = {
    "landsat": LANDSAT_ASSETS,
    "sentinel": SENTINEL_ASSETS,
    "naip": NAIP_ASSETS,
    "nz": NZ_ASSETS,
}

LANDSAT_ASSET_NORMS = {
    "coastal": [1, 1],
    "blue": [1, 1],
    "green": [1, 1],
    "red": [1, 1],
    "nir08": [1, 1],
    "swir16": [1, 1],
    "swir22": [1, 1],
}

NAIP_ASSET_NORMS = {
    "asset": [1, 1],
}

NZ_ASSET_NORMS = {
    "asset": [1, 1],
}

SENTINEL_ASSET_NORMS = {
    "aot": [1, 1],
    "blue": [1, 1],
    "coastal": [1, 1],
    "green": [1, 1],
    "nir": [1, 1],
    "nir08": [1, 1],
    "nir09": [1, 1],
    "red": [1, 1],
    "rededge1": [1, 1],
    "rededge2": [1, 1],
    "rededge3": [1, 1],
    "swir22": [1, 1],
}

NORM_LOOKUP = {
    "landsat": LANDSAT_ASSET_NORMS,
    "sentinel": SENTINEL_ASSET_NORMS,
    "naip": NAIP_ASSET_NORMS,
    "nz": NZ_ASSET_NORMS,
}


def token_bounds(bbox, shape, window):
    x_pixel_size = (bbox[2] - bbox[0]) / shape[1]
    y_pixel_size = (bbox[3] - bbox[1]) / shape[0]

    return (
        bbox[0] + window.col_off * x_pixel_size,
        bbox[1] + window.row_off * y_pixel_size,
        bbox[0] + (window.col_off + window.width) * x_pixel_size,
        bbox[1] + (window.row_off + window.height) * y_pixel_size,
    )


def get_chip(chipid, stac_item_name, chip_index_x, chip_index_y):
    print(chipid, stac_item_name, chip_index_x, chip_index_y)

    item = Item.from_file(stac_item_name)

    stac_item_shape = get_shape(item)
    platform = str(stac_item_name.name).split("-")[0].split("_")[0]
    assets = ASSET_LOOKUP[platform]
    norms = NORM_LOOKUP[platform]

    chip = []

    for key, asset in item.assets.items():
        if key not in assets:
            continue

        with rasterio.open(asset.href) as src:
            # Currently assume that different assets may be at different
            # resolutions, but are aligned and the gsd differs by an integer
            # multiplier.
            if stac_item_shape[0] % src.height:
                raise ValueError(
                    "Asset height {src.height} is not a multiple of highest resolution height {stac_item_shape[0]}"  # noqa: E501
                )

            if stac_item_shape[1] % src.width:
                raise ValueError(
                    "Asset width {src.width} is not a multiple of highest resolution height {stac_item_shape[1]}"  # noqa: E501
                )

            factor = stac_item_shape[0] / src.height
            if factor != 1:
                print(
                    f"Asset {key} is not at highest resolution using scaling factor of {factor}"  # noqa: E501
                )

            chip_window = Window(
                math.floor(chip_index_y / factor),
                math.floor(chip_index_x / factor),
                math.ceil(CHIP_SIZE / factor),
                math.ceil(CHIP_SIZE / factor),
            )
            print(f"Chip window for asset {key} is {chip_window}")
            bounds = token_bounds(item.bbox, src.shape, chip_window)
            token = (
                str(item.datetime.date()),  # Date
                *norms[key],  # Normalization (mean, std)
                *norms[key],  # Band def (central wavelength, bandwidth)
                src.transform[0],  # gsd
                *bounds,  # Lat/Lon bbox
                src.read(window=chip_window),
            )
            chip.append(token)

    return chip


def get_chips_from_index(index: Path, platform: str):
    print(f"Processing index {index}")

    with open(index) as src:
        index_data = json.load(src)
    print(f"Found {len(index_data)} chips to process")

    for row in index_data:
        stac_item_name = row.pop("stac_item_name")

        # Test with sentinel tile as it is multi-resolution
        if not stac_item_name.startswith(platform):
            continue
        chip = get_chip(stac_item_name=index.parent / "items" / stac_item_name, **row)

        print(stac_item_name)
        for token in chip:
            print(token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Clay v1 tokens.")
    parser.add_argument(
        "--index",
        type=Path,
        help="The index location, the items are assumed to be in an items subfolder from the index location.",  # noqa: E501
        required=True,
    )
    parser.add_argument(
        "--platform",
        type=str,
        help="Limit items from one platform. Should be one of: sentinel, landsat, nz, naip.",  # noqa: E501
        required=False,
    )
    args = parser.parse_args()

    assert args.index.exists()

    print(args.index)

    get_chips_from_index(args.index, args.platform)
