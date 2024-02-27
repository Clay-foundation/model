"""
Tiling module for multi-dimensional datacubes.

This script defines a tiling function for processing multi-dimensional imagery
stacks into smaller tiles, while filtering out tiles with high cloud coverage
or no-data pixels.

It includes functions to filter tiles based on cloud coverage and no-data pixels,
and a tiling function that generates smaller tiles from the input stack.
"""
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

NODATA = 0
TILE_SIZE = 512
PIXELS_PER_TILE = TILE_SIZE * TILE_SIZE
BAD_PIXEL_MAX_PERCENTAGE = 0.3
SCL_FILTER = [0, 1, 3, 8, 9, 10]


def filter_clouds_nodata(tile):
    """
    Filter tiles based on cloud coverage and no-data pixels.

    Args:
    - tile (xarray.Dataset): A subset of data representing a tile.

    Returns:
    - bool: True if the tile is approved, False if rejected.
    """
    # Check for nodata pixels
    if int(tile.sel(band="B02").isin([NODATA]).sum()):
        print("Too much no-data in B02")
        return False

    bands_to_check = ["vv", "vh", "dem"]
    for band in bands_to_check:
        if int(np.isnan(tile.sel(band=band)).sum()):
            print(f"Too much no-data in {band}")
            return False

    # Check for cloud coverage
    cloudy_pixel_count = int(tile.sel(band="SCL").isin(SCL_FILTER).sum())
    if cloudy_pixel_count / PIXELS_PER_TILE >= BAD_PIXEL_MAX_PERCENTAGE:
        print("Too much cloud coverage")
        return False

    return True  # If both conditions pass


def tiler(stack, date):
    """
    Function to tile a multi-dimensional imagery stack while filtering out
    tiles with high cloud coverage or no-data pixels.

    Args:
    - stack (xarray.Dataset): The input multi-dimensional imagery stack.
    - date (str): Date string yyyy-mm-dd
    """
    # Calculate the number of full tiles in x and y directions
    num_x_tiles = stack[0].x.size // TILE_SIZE
    num_y_tiles = stack[0].y.size // TILE_SIZE

    counter = 0
    # Iterate through each chunk of x and y dimensions and create tiles
    for y_idx in range(num_y_tiles):
        for x_idx in range(num_x_tiles):
            # Calculate the start and end indices for x and y dimensions
            # for the current tile
            x_start = x_idx * TILE_SIZE
            y_start = y_idx * TILE_SIZE
            x_end = x_start + TILE_SIZE
            y_end = y_start + TILE_SIZE

            # Select the subset of data for the current tile
            parts = [part[:, y_start:y_end, x_start:x_end] for part in stack]

            # Only concat here to save memory, it converts S2 data to float
            tile = xr.concat(parts, dim="band").rename("tile")

            counter += 1
            if counter % 100 == 0:
                print(f"Counted {counter} tiles")

            if not filter_clouds_nodata(tile):
                continue

            tile = tile.drop_sel(band="SCL")

            bounds = tile.rio.transform_bounds("EPSG:4326")

            yield {
                "pixels": tile.to_numpy(),
                "date": date,
                "lat": bounds[1] + (bounds[1] - bounds[3]) / 2,
                "lon": bounds[0] + (bounds[0] - bounds[2]) / 2,
            }
