"""
Module to create a Clay v1 index file from a folder of STAC items

The STAC items are expected to have the proj extension to extract the shape
of the assets in the STAC item.

The STAC items are expected to be in an ./items subfolder from the working
directory. The index is written to the working directory.
"""

import argparse
import json
from pathlib import Path

from pystac import Item

CHIP_SIZE = 256


def get_shape(item: Item) -> list:
    shape = None
    if "proj:shape" in item.properties:
        shape = item.properties["proj:shape"]
    else:
        for asset in item.assets.values():
            if "proj:shape" not in asset.extra_fields:
                continue
            if shape is None or shape[0] < asset.extra_fields["proj:shape"][0]:
                shape = asset.extra_fields["proj:shape"]
    return shape


def create_index(wd: Path):
    index = []
    chipid = 0
    for item_path in args.wd.glob("items/*.json"):
        print(item_path)
        item = Item.from_file(item_path)

        shape = get_shape(item)

        print("Shape", shape)
        for y in range(0, shape[1], CHIP_SIZE):
            for x in range(0, shape[0], CHIP_SIZE):
                row = {
                    "chipid": chipid,
                    "stac_item_name": item_path.name,
                    "chip_index_x": x,
                    "chip_index_y": y,
                }
                print(row)
                index.append(row)
                chipid += 1

    with open(wd / "clay-v1-index.json", "w") as dst:
        json.dump(index, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Clay v1 index.")
    parser.add_argument(
        "--wd",
        type=Path,
        help="The working directory. Items are expected to be in an ./items subdirectory",  # noqa: E501
        required=True,
    )
    args = parser.parse_args()

    assert args.wd.exists()

    create_index(args.wd)
