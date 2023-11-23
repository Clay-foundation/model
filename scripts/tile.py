"""
Tiling module for multi-dimensional datacubes.

This script defines a tiling function for processing multi-dimensional imagery
stacks into smaller tiles, while filtering out tiles with high cloud coverage
or no-data pixels.

It includes functions to filter tiles based on cloud coverage and no-data pixels,
and a tiling function that generates smaller tiles from the input stack.
"""
import os
import subprocess
import tempfile

import numpy as np
import rasterio
import rioxarray  # noqa: F401
from rasterio.enums import ColorInterp

NODATA = 0
TILE_SIZE = 256
PIXELS_PER_TILE = TILE_SIZE * TILE_SIZE
BAD_PIXEL_MAX_PERCENTAGE = 0.9
SCL_FILTER = [0, 1, 3, 8, 9, 10]
EPSILON = 0.1
VERSION = "01"


def filter_clouds_nodata(tile):
    """
    Filter tiles based on cloud coverage and no-data pixels.

    Args:
    - tile (xarray.Dataset): A subset of data representing a tile.

    Returns:
    - bool: True if the tile is approved, False if rejected.
    """
    # Check for nodata pixels
    nodata_pixel_count = int(tile.sel(band="B02").isin([NODATA]).sum())
    if nodata_pixel_count:
        print("Too much no-data")
        return False

    # Check for cloud coverage
    cloudy_pixel_count = int(tile.sel(band="SCL").isin(SCL_FILTER).sum())
    if cloudy_pixel_count / PIXELS_PER_TILE >= BAD_PIXEL_MAX_PERCENTAGE:
        print("Too much cloud coverage")
        return False

    return True  # If both conditions pass


def tiler(stack, date, mgrs, bucket):
    """
    Function to tile a multi-dimensional imagery stack while filtering out
    tiles with high cloud coverage or no-data pixels.

    Args:
    - stack (xarray.Dataset): The input multi-dimensional imagery stack.
    - date (str): Date string yyyy-mm-dd
    - mgrs (str): MGRS Tile id
    - bucket(str): AWS S3 bucket to write tiles to
    """
    # Calculate the number of full tiles in x and y directions
    num_x_tiles = stack.x.size // TILE_SIZE
    num_y_tiles = stack.y.size // TILE_SIZE

    bucket = os.environ.get("TARGET_BUCKET", "whis-imagery")

    counter = 0
    with tempfile.TemporaryDirectory() as dir:
        print("Writing tempfiles to ", dir)
        # Iterate through each chunk of x and y dimensions and create tiles
        for y_idx in range(num_y_tiles):
            for x_idx in range(num_x_tiles):
                counter += 1
                print(f"Counted {counter} tiles")

                # Calculate the start and end indices for x and y dimensions
                # for the current tile
                x_start = x_idx * TILE_SIZE
                y_start = y_idx * TILE_SIZE
                x_end = x_start + TILE_SIZE
                y_end = y_start + TILE_SIZE

                # Select the subset of data for the current tile
                tile = stack.sel(
                    x=slice(
                        stack.x.values[x_start],
                        stack.x.values[x_end]
                        + np.sign(stack.rio.transform()[4]) * EPSILON,
                    ),
                    y=slice(
                        stack.y.values[y_start],
                        stack.y.values[y_end]
                        + np.sign(stack.rio.transform()[0]) * EPSILON,
                    ),
                )

                if not filter_clouds_nodata(tile):
                    continue

                tile = tile.drop_sel(band="SCL")

                # Track band names and color interpretation
                tile.attrs["long_name"] = [str(x.values) for x in tile.band]
                color = [ColorInterp.blue, ColorInterp.green, ColorInterp.red] + [
                    ColorInterp.gray
                ] * (len(tile.band) - 3)

                # Write tile to tempdir
                name = f"{dir}/claytile-{mgrs}-{date}-{VERSION}-{counter}.tif"
                tile.rio.to_raster(name, compress="deflate")

                with rasterio.open(name, "r+") as rst:
                    rst.colorinterp = color

        print(f"Syncing {dir} with s3://{bucket}/{VERSION}/{mgrs}/{date}")
        subprocess.run(
            ["aws", "s3", "sync", dir, f"s3://{bucket}/{VERSION}/{mgrs}/{date}"],
            check=True,
        )
