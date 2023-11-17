"""
Tiling module for multi-dimensional datacubes.

This script defines a tiling function for processing multi-dimensional imagery stacks into smaller tiles,
while filtering out tiles with high cloud coverage or no-data pixels.

It includes functions to filter tiles based on cloud coverage and no-data pixels,
and a tiling function that generates smaller tiles from the input stack.
"""

import numpy as np
import xarray as xr

NODATA = 0
TILE_SIZE = 256
PIXELS_PER_TILE = TILE_SIZE * TILE_SIZE
BAD_PIXEL_MAX_PERCENTAGE = 1.0
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
    nodata_pixel_count = int(tile.B02.isin([NODATA]).sum())
    if nodata_pixel_count:
        print("Too much no-data")
        return False

    # Check for cloud coverage
    cloudy_pixel_count = int(tile.SCL.isin(SCL_FILTER).sum())
    if cloudy_pixel_count / PIXELS_PER_TILE > BAD_PIXEL_MAX_PERCENTAGE:
        print("Too much cloud coverage")
        return False

    return True  # If both conditions pass


def tiler(stack):
    """
    Function to tile a multi-dimensional imagery stack while filtering out tiles with high cloud coverage or no-data pixels.

    Args:
    - stack (xarray.Dataset): The input multi-dimensional imagery stack.

    Returns:
    - list: A list containing approved tiles with specified dimensions.
    """
    # Calculate the number of full tiles in x and y directions
    num_x_tiles = stack.x.size // TILE_SIZE
    num_y_tiles = stack.y.size // TILE_SIZE

    # Calculate the remaining sizes in x and y directions
    remainder_x = stack.x.size % TILE_SIZE
    remainder_y = stack.y.size % TILE_SIZE

    # Create a list to hold the tiles
    tiles = []

    # Counter for tiles
    tile_count = 0

    # Counter for bad tiles
    # bad_tile_count = 0

    # Iterate through each chunk of x and y dimensions and create tiles
    for y_idx in range(num_y_tiles + 1 if remainder_y > 0 else num_y_tiles):
        for x_idx in range(num_x_tiles + 1 if remainder_x > 0 else num_x_tiles):
            # Calculate the start and end indices for x and y dimensions for the current tile
            x_start = x_idx * TILE_SIZE
            y_start = y_idx * TILE_SIZE
            x_end = min((x_idx + 1) * TILE_SIZE, stack.x.size)
            y_end = min((y_idx + 1) * TILE_SIZE, stack.y.size)
            
            # Select the subset of data for the current tile
            tile = stack.sel(x=slice(stack.x.values[x_start], stack.x.values[x_end - 1]), y=slice(stack.y.values[y_start], stack.y.values[y_end - 1]))
            tile_spatial_dims = tuple(tile.dims[d] for d in ['x', 'y'])
            if tile_spatial_dims[0] == TILE_SIZE and tile_spatial_dims[1] == TILE_SIZE:
                tile_count=tile_count+1
                print("Tile size: ", tuple(tile.dims[d] for d in ['x', 'y']), "; tile count: ", tile_count)
                # Check for clouds and nodata
                approval = filter_clouds_nodata(tile)
                if approval == True:
                    # Append the tile to the list 
                    tiles.append(tile)
                else:
                    pass # bad_tile_count = bad_tile_count + 1
            else:
                pass
    # print(f"{bad_tile_count} tiles removed due to clouds or nodata")
    # 'tiles' now contains tiles with 256x256 pixels for x and y
    return tiles
