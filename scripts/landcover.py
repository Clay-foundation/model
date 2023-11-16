from pathlib import Path

import click
import fiona
import geopandas
import numpy
import pandas
import rasterio
from fiona.crs import CRS
from rasterio.windows import from_bounds
from shapely.geometry import shape

WGS84 = CRS.from_epsg(4326)
NODATA = 0
WATER = 80
WATER_LOWER_TH = 0.2
WATER_UPPER_TH = 0.7
RANDOM_SEED = 42
CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and Ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
SCHEMA = {
    "geometry": "MultiPolygon",
    "properties": {
        "name": "str",
        "count": "int",
        "Tree cover": "int",
        "Shrubland": "int",
        "Grassland": "int",
        "Cropland": "int",
        "Built-up": "int",
        "Bare / sparse vegetation": "int",
        "Snow and Ice": "int",
        "Permanent water bodies": "int",
        "Herbaceous wetland": "int",
        "Mangroves": "int",
        "Moss and lichen": "int",
    },
}


@click.command()
@click.option(
    "--wd",
    required=True,
    type=str,
)
@click.option(
    "--worldcover",
    required=True,
    type=str,
)
@click.option(
    "--mgrs",
    required=True,
    type=str,
)
def process(wd, worldcover, mgrs):
    """
    Run statistics and sampling.
    """
    compute_stats(wd, worldcover, mgrs)
    sample(wd)


def compute_stats(wd, worldcover, mgrs):
    """
    Compute statistics of Worldcover data over MGRS tiles.
    """
    result = []
    with rasterio.open(worldcover) as cover:
        with fiona.open(mgrs, "r") as tiles:
            assert cover.crs.to_epsg() == tiles.crs.to_epsg()
            for tile in tiles:
                print(tile.properties.get("Name"))
                # Split polygons in parts, mgrs tiles at the dateline
                # are split into two parts in the mgrs source file.
                parts = shape(tile["geometry"]).geoms
                pixels = []
                for polygon in parts:
                    bounds = from_bounds(*polygon.bounds, cover.transform)
                    pixels.append(
                        cover.read(
                            1,
                            window=bounds,
                        ).ravel()
                    )
                pixels = numpy.hstack(pixels)

                pixels = pixels[pixels != NODATA]

                if not pixels.size:
                    continue
                elif numpy.all(pixels == WATER):
                    continue
                else:
                    props = {}
                    for key, classname in CLASSES.items():
                        props[str(classname)] = int(numpy.sum(pixels == key))
                    props["name"] = tile.properties.get("Name")
                    props["count"] = int(len(numpy.unique(pixels)))

                    result.append(
                        {
                            "geometry": dict(tile["geometry"]),
                            "properties": props,
                        }
                    )

    with fiona.open(
        Path(wd, "mgrs_stats.fgb"),
        "w",
        driver="FlatGeobuf",
        crs=WGS84,
        schema=SCHEMA,
    ) as colxn:
        colxn.writerecords(result)


def split_highest(data, column, size, pool=1000, seed=RANDOM_SEED):
    """
    Split highest values of a column from a dataframe.
    """
    data.sort_values(column, ascending=False, inplace=True)
    return data[:pool].sample(size, random_state=seed)


def percentages(data):
    """
    Normalize all numerical columns to percentages
    """
    data_num = data.select_dtypes(include="number")
    data_norm = data_num.div(data_num.sum(axis=1), axis=0)
    data[data_norm.columns] = data_norm

    return data


def sample(wd):
    """
    Sample the mgrs tiles based on landcover statistics.

    Target: ~1000 tiles
    Set very small counts to zero. Exclude high latitudes.
    200 samples from the 2000 most diverse
    50 samples from the 1000 highest for all other categories except water
    100 samples from all tiles with water between 30% an 70% (making sure we
    capture some, but exclude only purely water so we catch coasts)
    """
    data = geopandas.read_file(Path(wd, "mgrs_stats.fgb"))

    data_norm = percentages(data.loc[:, data.columns != "count"])
    data[data_norm.columns] = data_norm

    diversity = split_highest(data, "count", 200, 2000)
    urban = split_highest(data, "Built-up", 200)
    wetland = split_highest(data, "Herbaceous wetland", 50)
    mangroves = split_highest(data, "Mangroves", 50)
    moss = split_highest(data, "Moss and lichen", 50)
    cropland = split_highest(data, "Cropland", 50)
    trees = split_highest(data, "Tree cover", 50)
    shrubland = split_highest(data, "Shrubland", 50)
    grassland = split_highest(data, "Grassland", 50)
    bare = split_highest(data, "Bare / sparse vegetation", 50)
    snow = split_highest(data, "Snow and Ice", 50)

    selector = numpy.logical_and(
        data["Permanent water bodies"] > WATER_LOWER_TH,
        data["Permanent water bodies"] < WATER_UPPER_TH,
    )
    water = data[selector].sample(100, random_state=RANDOM_SEED)

    result = pandas.concat(
        [
            diversity,
            urban,
            wetland,
            mangroves,
            moss,
            cropland,
            trees,
            shrubland,
            grassland,
            bare,
            snow,
            water,
        ]
    )

    result = result.drop_duplicates(subset=["name"])

    result.to_file(Path(wd, "mgrs_sample.geojson", driver="GeoJSON"))


if __name__ == "__main__":
    process()
