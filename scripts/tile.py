"""
Module to create tiles of 256x256 pixels from a Xarray dataset.
"""

import numpy as np
import xarray as xr


def tiler(stack):
    # Define tile size
    tile_size = 256

    # Calculate the number of full tiles in x and y directions
    num_x_tiles = stack.x.size // tile_size
    num_y_tiles = stack.y.size // tile_size

    # Calculate the remaining sizes in x and y directions
    remainder_x = stack.x.size % tile_size
    remainder_y = stack.y.size % tile_size

    # Create a list to hold the tiles
    tiles = []

    # Iterate through each chunk of x and y dimensions and create tiles
    for y_idx in range(num_y_tiles + 1 if remainder_y > 0 else num_y_tiles):
        for x_idx in range(num_x_tiles + 1 if remainder_x > 0 else num_x_tiles):
            # Calculate the start and end indices for x and y dimensions for the current tile
            x_start = x_idx * tile_size
            y_start = y_idx * tile_size
            x_end = min((x_idx + 1) * tile_size, stack.x.size)
            y_end = min((y_idx + 1) * tile_size, stack.y.size)
            
            # Select the subset of data for the current tile
            tile = stack.sel(x=slice(stack.x.values[x_start], stack.x.values[x_end - 1]), y=slice(stack.y.values[y_start], stack.y.values[y_end - 1]))
            tile_spatial_dims = tuple(tile.dims[d] for d in ['x', 'y'])
            if tile_spatial_dims[0] == 256 and tile_spatial_dims[1] == 256:
                print("Tile size: ", tuple(tile.dims[d] for d in ['x', 'y']))
                # Append the tile to the list
                tiles.append(tile)
            else:
                pass
    
    # 'tiles' now contains tiles with 256x256 pixels for x and y
    return tiles
