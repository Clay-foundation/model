"""
Module to retrieve Sentinel-2 tiles.

This approach uses STAC for scene discovery based on a date range and a location.
The least cloudy scene is retrieved as a whole, and passed to a tiler. The tiler
will create tiles of 256x256 pixels at different resolutions, track their boundaries,
and centroids.
"""
import backoff
import click
import numpy
import planetary_computer
import pystac_client
import stackstac
from pystac_client.exceptions import APIError
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
from requests import ConnectionError, HTTPError

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
SIZE = 10980
NODATA = 0
# SCL classes.
# 0: NO_DATA
# 1: SATURATED_OR_DEFECTIVE
# 2: DARK_AREA_PIXELS
# 3: CLOUD_SHADOWS
# 4: VEGETATION
# 5: NOT_VEGETATED
# 6: WATER
# 7: UNCLASSIFIED
# 8: CLOUD_MEDIUM_PROBABILITY
# 9: CLOUD_HIGH_PROBABILITY
# 10: THIN_CIRRUS
# 11: SNOW
SCL_FILTER = [0, 1, 3, 8, 9, 10]
TILE_SIZE = 256
PIXELS_PER_TILE = TILE_SIZE * TILE_SIZE
BAD_PIXEL_MAX_PERCENTAGE = 1.0
RESOLUTION = 10
RESRES = [20, 40, 80, 160, 300]
NROFTILES = 40
TILESFIT = NROFTILES * TILE_SIZE
COLLECTION = "sentinel-2-l2a"
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"


@backoff.on_exception(
    backoff.expo,
    (HTTPError, ConnectionError, RasterioIOError, APIError),
    max_tries=10,
    jitter=backoff.full_jitter,
    raise_on_giveup=True,
)
def search(start="2020-03-01", end="2020-06-01", lon=-105.78, lat=35.79, max_items=1):
    """
    Search Sentinel-2 STAC items on the MS planetary computer.

    Looks for the least cloudy scene in a time window over a location.
    """
    catalog = pystac_client.Client.open(
        STAC_ENDPOINT,
        modifier=planetary_computer.sign_inplace,
    )
    # Least cloudy scene for the time period.
    items = catalog.search(
        intersects=dict(type="Point", coordinates=[lon, lat]),
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        sortby="eo:cloud_cover",
        max_items=max_items,
    ).item_collection()

    return items


@backoff.on_exception(
    backoff.expo,
    (HTTPError, ConnectionError, RasterioIOError, APIError),
    max_tries=10,
    jitter=backoff.full_jitter,
    raise_on_giveup=True,
)
def retrieve(items):
    """
    Download the data.
    """
    stack = stackstac.stack(
        items, resolution=RESOLUTION, assets=BANDS, dtype="uint16", fill_value=NODATA
    )

    stack = stack.compute()

    return stack


def bbox_centroid(bounds, epsg):
    """
    Compute lat/lon centroid of input bounds.
    """
    bounds_4326 = transform_bounds(
        {"init": f"EPSG:{epsg}"},
        {"init": "EPSG:4326"},
        *bounds,
    )
    lon = bounds_4326[0] + (bounds_4326[2] - bounds_4326[0]) / 2
    lat = bounds_4326[1] + (bounds_4326[3] - bounds_4326[1]) / 2

    return numpy.array([lat, lon])


def tiler(stack, levels=6):
    """
    Tile the array and filter by nodata and clouds.
    """
    tiles = []
    bounds = []
    scales = []
    centroids = []

    for level in range(levels):
        data = stack.squeeze().transpose("x", "y", "band")
        scale = RESOLUTION * 2**level
        nroftiles = int(NROFTILES / 2**level)

        if level == 0:
            data = data[:TILESFIT, :TILESFIT, :]
        else:
            data = data.coarsen(x=2**level, y=2**level, boundary="trim").mean()

        print("Working with", dict(data=data.shape, scale=scale, nroftiles=nroftiles))

        for i in range(nroftiles):
            for j in range(nroftiles):
                tile = data[
                    i * TILE_SIZE : (i + 1) * TILE_SIZE,
                    j * TILE_SIZE : (j + 1) * TILE_SIZE,
                    :,
                ]

                # Skip if the tile has nodata pixels
                nodata_pixel_count = int(tile.sel(band="B02").isin([NODATA]).sum())
                if nodata_pixel_count:
                    continue

                # Cloud and bad pixel percentage based on the SCL band
                cloudy_pixel_count = int(tile.sel(band="SCL").isin(SCL_FILTER).sum())
                if cloudy_pixel_count / PIXELS_PER_TILE > BAD_PIXEL_MAX_PERCENTAGE:
                    continue

                # Append only spectral bands to tiles list
                tiles.append(tile.sel(band=BANDS[:10]).to_numpy())

                # Track bounds, centroids and scale
                tile_bounds = numpy.array(
                    [
                        stack.attrs["spec"].bounds[0] + i * TILE_SIZE * scale,
                        stack.attrs["spec"].bounds[1] + j * TILE_SIZE * scale,
                        stack.attrs["spec"].bounds[0] + (i + 1) * TILE_SIZE * scale,
                        stack.attrs["spec"].bounds[1] + (j + 1) * TILE_SIZE * scale,
                    ]
                )
                bounds.append(tile_bounds)

                centroids.append(bbox_centroid(tile_bounds, stack.attrs["spec"].epsg))

                scales.append(scale)

    return tiles, bounds, centroids, scales


def plot(tile):
    """
    Plot a tile in RGB.
    """
    from matplotlib import pyplot as plt

    rgb = (
        numpy.array([tile[:, :, 2], tile[:, :, 1], tile[:, :, 0]]).clip(0, 3000) / 3000
    )
    plt.imshow(rgb.transpose(1, 2, 0))
    plt.show()


@click.command()
@click.option(
    "--start",
    required=True,
    type=str,
    help="Start date in YYYY-MM-DD",
)
@click.option(
    "--end",
    required=True,
    type=str,
    help="End date in YYYY-MM-DD",
)
@click.option(
    "--lon",
    required=True,
    type=float,
    help="Longitude to find tiles",
)
@click.option(
    "--lat",
    required=True,
    type=float,
    help="Latitude to find tiles",
)
def process(start="2020-03-01", end="2020-06-01", lon=-105.78, lat=35.79):
    """
    Data preparation pipeline.
    """
    print(start, end, lon, lat)
    items = search(start, end, lon, lat)
    stack = retrieve(items)
    tiles, bounds, centroids, scales = tiler(stack)
    print(f"Storing {len(tiles)} tiles")
    # TODO: Make this an upload to S3.
    numpy.savez_compressed(
        f"/datadisk/clay/{stack.id.to_numpy()[0]}.npz",
        id=stack.id.to_numpy()[0],
        time=stack.time.to_numpy()[0],
        epsg=int(stack.epsg),
        bounds=numpy.stack(bounds),
        scales=numpy.array(scales),
        centroids=numpy.stack(centroids),
        tiles=numpy.stack(tiles),
    )


if __name__ == "__main__":
    process()
