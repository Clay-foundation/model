#!/usr/bin/env python3
"""
STAC Data Processing Script

This Python script processes Sentinel-2, Sentinel-1, and Copernicus DEM
(Digital Elevation Model) data. It utilizes Microsoft's Planetary Computer API
for data retrieval and manipulation.

Constants:
- STAC_API: Planetary Computer API endpoint
- S2_BANDS: Bands used in Sentinel-2 data processing

Functions:
- get_surrounding_days(reference, interval_days):
      Get the week range for a given date.
- search_sentinel2(date_range, aoi, cloud_cover_percentage, nodata_pixel_percentage):
      Search for Sentinel-2 items within a given date range and area of interest.
- search_sentinel1(bbox, catalog, date_range):
      Search for Sentinel-1 items within a given bounding box, STAC catalog,
      and date range.
- search_dem(bbox, catalog):
      Search for DEM items within a given bounding box.
- make_datasets(s2_item, s1_items, dem_items, resolution):
      Create xarray Datasets for Sentinel-2, Sentinel-1, and DEM data.
- process(aoi, year, resolution, cloud_cover_percentage, nodata_pixel_percentage):
      Process Sentinel-2, Sentinel-1, and DEM data for a specified time range,
      area of interest, and resolution.
"""
import random
from datetime import timedelta

import click
import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import stackstac
import xarray as xr
from pystac import ItemCollection
from shapely.geometry import box
from tile import tiler

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
SPATIAL_RESOLUTION = 10
CLOUD_COVER_PERCENTAGE = 50
NODATA_PIXEL_PERCENTAGE = 20
NODATA = 0
S1_MATCH_ATTEMPTS = 20
DATES_PER_LOCATION = 3


def get_surrounding_days(reference, interval_days):
    """
    Get the days surrounding the input date.

    Parameters:
    - reference (datetime): The reference datetime.
    - interval_days (int): The number of days to search ahead and back

    Returns:
    - str: A string representing the start and end dates of the date interval in the
        format 'start_date/end_date'.
    """
    start = reference - timedelta(days=interval_days)
    end = reference + timedelta(days=interval_days)
    return f"{start.date()}/{end.date()}"


def search_sentinel2(
    catalog, date_range, aoi, cloud_cover_percentage, nodata_pixel_percentage, index=0
):
    """
    Search for Sentinel-2 items within a given date range and area of interest (AOI)
    with specified conditions.

    Parameters:
    - catalog (pystac.Catalog): STAC catalog containing Sentinel-2 items.
    - date_range (str): The date range in the format 'start_date/end_date'.
    - aoi (shapely.geometry.base.BaseGeometry): Geometry object for an Area of
        Interest (AOI).
    - cloud_cover_percentage (int): Maximum acceptable cloud cover percentage
        for Sentinel-2 images.
    - nodata_pixel_percentage (int): Maximum acceptable percentage of nodata
        pixels in Sentinel-2 images.
    - index: Which of the found scenes to select

    Returns:
    - tuple: A tuple containing the STAC catalog, Sentinel-2 items, and the
        bounding box (bbox)

    Note:
    The function filters Sentinel-2 items based on the specified conditions
    such as geometry, date, cloud cover, and nodata pixel percentage. Only one
    result with the least cloud cover will be returned. The result is returned
    as a tuple containing the STAC catalog, Sentinel-2 items, the bounding box
    of the first item, and an EPSG code for the coordinate reference system.
    """
    search: pystac_client.item_search.ItemSearch = catalog.search(
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {
                    "op": "s_intersects",
                    "args": [{"property": "geometry"}, aoi.centroid.__geo_interface__],
                },
                {"op": "anyinteracts", "args": [{"property": "datetime"}, date_range]},
                {"op": "=", "args": [{"property": "collection"}, "sentinel-2-l2a"]},
                {
                    "op": "<=",
                    "args": [{"property": "eo:cloud_cover"}, cloud_cover_percentage],
                },
                {
                    "op": "<=",
                    "args": [
                        {"property": "s2:nodata_pixel_percentage"},
                        nodata_pixel_percentage,
                    ],
                },
            ],
        },
    )

    s2_items = search.item_collection()
    print(f"Found {len(s2_items)} Sentinel-2 items")
    if not len(s2_items):
        return None, None

    s2_items_gdf = gpd.GeoDataFrame.from_features(s2_items.to_dict())

    least_clouds = s2_items_gdf.sort_values(by=["eo:cloud_cover"], ascending=True).iloc[
        index
    ]

    # Get the datetime for the filtered Sentinel 2 dataframe
    # containing the least nodata and least cloudy scene
    for item in s2_items:
        if item.properties["datetime"] == least_clouds["datetime"]:
            s2_item = item
            break

    bbox = least_clouds.geometry.bounds

    epsg = s2_item.properties["proj:epsg"]
    print("EPSG code based on Sentinel-2 item: ", epsg)

    return s2_item, bbox


def search_sentinel1(bbox, catalog, date_range):
    """
    Search for Sentinel-1 items within a given bounding box (bbox), STAC
    catalog, and date range.

    Parameters:
    - bbox (tuple): Bounding box coordinates in the format
        (minx, miny, maxx, maxy).
    - catalog (pystac.Catalog): STAC catalog containing Sentinel-1 items.
    - date_range (str): The date range in the format 'start_date/end_date'.

    Returns:
    - pystac.Collection: A collection of Sentinel-1 items filtered by specified
        conditions.

    Note:
    This function retrieves Sentinel-1 items from the catalog that intersect
    with the given bounding box and fall within the provided time window. The
    function filters items based on orbit state and returns the collection of
    Sentinel-1 items that meet the defined criteria.
    """
    # Create poly geom object from the bbox
    geom_bbox = box(*bbox)

    search: pystac_client.item_search.ItemSearch = catalog.search(
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {
                    "op": "s_intersects",
                    "args": [{"property": "geometry"}, geom_bbox.__geo_interface__],
                },
                {"op": "anyinteracts", "args": [{"property": "datetime"}, date_range]},
                {"op": "=", "args": [{"property": "collection"}, "sentinel-1-rtc"]},
            ],
        },
    )
    s1_items = search.item_collection()
    print(f"Found {len(s1_items)} Sentinel-1 items")

    if not len(s1_items):
        return
    else:
        # Add id as property to persist in gdf
        for item in s1_items:
            item.properties["id"] = item.id

        # Try to find enough scenes with same orbit that fully overlap
        # the S2 bbox.
        s1_gdf = gpd.GeoDataFrame.from_features(s1_items)
        s1_gdf["overlap"] = s1_gdf.intersection(box(*bbox)).area
        s1_gdf = s1_gdf.sort_values(by="overlap", ascending=False)

        most_overlap_orbit = s1_gdf.iloc[0]["sat:orbit_state"]
        print("Most overlapped orbit: ", most_overlap_orbit)
        selected_item_ids = []
        intersection = None
        orbit = None
        for index, row in s1_gdf.iterrows():
            orbit = row["sat:orbit_state"]
            if intersection is None and orbit == most_overlap_orbit:
                intersection = row.geometry
                selected_item_ids.append(row.id)
                intersection = intersection.intersection(row.geometry)
            elif orbit == most_overlap_orbit and not intersection.covers(geom_bbox):
                intersection = row.geometry
                selected_item_ids.append(row.id)
                intersection = intersection.intersection(row.geometry)
            elif orbit == most_overlap_orbit and intersection.covers(geom_bbox):
                # Stop adding scenes when the bbox is fully covered.
                break
            else:
                pass

        s1_items = ItemCollection(
            [item for item in s1_items if item.id in selected_item_ids]
        )

        return s1_items


def search_dem(bbox, catalog):
    """
    Search for Copernicus Digital Elevation Model (DEM) items within a given
    bounding box (bbox), STAC catalog, and Sentinel-2 items.

    Parameters:
    - bbox (tuple): Bounding box coordinates in the format
        (minx, miny, maxx, maxy).
    - catalog (pystac.Catalog): STAC catalog containing DEM items.

    Returns:
    - pystac.Collection: A collection of Digital Elevation Model (DEM) items
        filtered by specified conditions.
    """
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    dem_items = search.item_collection()
    print(f"Found {len(dem_items)} DEM items")

    return dem_items


def make_datasets(s2_items, s1_items, dem_items, resolution):
    """
    Create xarray Datasets for Sentinel-2, Sentinel-1, and Copernicus DEM
    data.

    Parameters:
    - s2_items (list): List of Sentinel-2 items.
    - s1_items (list): List of Sentinel-1 items.
    - dem_items (list): List of DEM items.
    - resolution (int): Spatial resolution.

    Returns:
    - tuple: A tuple containing xarray Datasets for Sentinel-2, Sentinel-1,
        and Copernicus DEM.
    """
    da_sen2: xr.DataArray = stackstac.stack(
        items=s2_items,
        assets=S2_BANDS,
        resolution=resolution,
        dtype=np.uint16,
        fill_value=0,
    )

    da_sen1: xr.DataArray = stackstac.stack(
        items=s1_items,
        assets=["vh", "vv"],
        epsg=int(da_sen2.epsg),
        bounds=da_sen2.spec.bounds,
        resolution=resolution,
        dtype=np.float32,
        fill_value=np.nan,
    )

    da_dem: xr.DataArray = stackstac.stack(
        items=dem_items,
        epsg=int(da_sen2.epsg),
        bounds=da_sen2.spec.bounds,
        resolution=resolution,
        dtype=np.float32,
        fill_value=np.nan,
    )

    da_sen1: xr.DataArray = stackstac.mosaic(da_sen1, dim="time")

    da_sen1 = da_sen1.drop_vars(
        [var for var in da_sen1.coords if var not in da_sen1.dims]
    )

    da_sen2 = da_sen2.drop_vars(
        [var for var in da_sen2.coords if var not in da_sen2.dims]
    ).squeeze()

    del da_sen2.coords["time"]

    da_dem: xr.DataArray = stackstac.mosaic(da_dem, dim="time").assign_coords(
        {"band": ["dem"]}
    )

    da_dem = da_dem.drop_vars([var for var in da_dem.coords if var not in da_dem.dims])

    return [da_sen2, da_sen1, da_dem]


def process(
    aoi,
    date_range,
    resolution,
    cloud_cover_percentage,
    nodata_pixel_percentage,
):
    """
    Process Sentinel-2, Sentinel-1, and Copernicus DEM data for a specified
    date_range, area of interest (AOI), resolution, EPSG code, cloud cover
    percentage, and nodata pixel percentage.

    Parameters:
    - aoi (shapely.geometry.base.BaseGeometry): Geometry object for an Area of
        Interest (AOI).
    - date_range (str): Date range string to pass to the catalog search.
    - resolution (int): Spatial resolution.
    - cloud_cover_percentage (int): Maximum acceptable cloud cover percentage
        for Sentinel-2 images.
    - nodata_pixel_percentage (int): Maximum acceptable percentage of nodata
        pixels in Sentinel-2 images.

    Returns:
    - xr.Dataset: Merged xarray Dataset containing processed data.
    """
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)

    for i in range(S1_MATCH_ATTEMPTS):
        s2_item, bbox = search_sentinel2(
            catalog,
            date_range,
            aoi,
            cloud_cover_percentage,
            nodata_pixel_percentage,
            index=i,
        )
        if not s2_item:
            continue

        surrounding_days = get_surrounding_days(s2_item.datetime, interval_days=3)
        print("Searching S1 in date range", surrounding_days)

        s1_items = search_sentinel1(bbox, catalog, surrounding_days)

        if s1_items:
            break

    if i == S1_MATCH_ATTEMPTS - 1:
        print(
            "No match for S1 scenes found for date range "
            f"{date_range} after {S1_MATCH_ATTEMPTS} attempts."
        )
        return None, None

    dem_items = search_dem(bbox, catalog)

    date = s2_item.properties["datetime"][:10]

    result = make_datasets(
        s2_item,
        s1_items,
        dem_items,
        resolution,
    )

    if 0 in (dat.shape[0] for dat in result):
        print("S2/S1 pixel coverages do not overlap although bounds do")
        return None, None

    return date, result


def convert_attrs_and_coords_objects_to_str(data):
    """
    Convert attributes and coordinates that are objects to
    strings.

    This is required for storing the xarray in netcdf.
    """
    for key, coord in data.coords.items():
        if coord.dtype == "object":
            data.coords[key] = str(coord.values)

    for key, attr in data.attrs.items():
        data.attrs[key] = str(attr)

    for key, var in data.variables.items():
        var.attrs = {}


@click.command()
@click.option(
    "--sample",
    required=False,
    default="https://clay-mgrs-samples.s3.amazonaws.com/mgrs_sample.fgb",
    help="Location of MGRS tile sample",
)
@click.option(
    "--index",
    required=False,
    default=0,
    help="Index of MGRS tile from sample file that should be processed",
)
@click.option(
    "--bucket",
    required=False,
    default="",
    help="Specify the bucket for where to write the data.",
)
@click.option(
    "--subset",
    required=False,
    default="",
    help="For debugging, subset the MGRS tile data to this pixel window."
    "Expects a comma separated string of 4 integers.",
    type=str,
)
@click.option(
    "--localpath",
    required=False,
    default=None,
    help="If specified, this path will be used to write the tiles locally"
    "Otherwise a temp dir will be used.",
)
@click.option(
    "--dateranges",
    required=False,
    default="",
    type=str,
    help="Comma separated list of date ranges, each provided as YYYY-MM-DD/YYYY-MM-DD.",
)
def main(sample, index, subset, bucket, localpath, dateranges):
    index = int(index)
    tiles = gpd.read_file(sample)
    tile = tiles.iloc[index]
    mgrs = tile["name"]

    print(f"Starting algorithm for MGRS tile {tile['name']} with index {index}")

    if subset:
        subset = [int(dat) for dat in subset.split(",")]

    if dateranges:
        date_ranges = dateranges.split(",")
    else:
        # Shuffle years, use index as seed for reproducibility but no
        # to have the same shuffle every time.
        date_ranges = [
            f"{year}-01-01/{year}-12-31"
            for year in (2017, 2018, 2019, 2020, 2021, 2022, 2023)
        ]
        random.seed(index)
        random.shuffle(date_ranges)

    match_count = 0
    for date_range in date_ranges:
        print(f"Processing data for date range {date_range}")
        date, pixels = process(
            tile.geometry,
            date_range,
            SPATIAL_RESOLUTION,
            CLOUD_COVER_PERCENTAGE,
            NODATA_PIXEL_PERCENTAGE,
        )
        if date is None:
            continue
        else:
            match_count += 1

        if subset:
            print(f"Subsetting to {subset}")
            pixels = [
                part[:, subset[1] : subset[3], subset[0] : subset[2]] for part in pixels
            ]

        pixels = [part.compute() for part in pixels]

        tiler(pixels, date, mgrs, bucket, localpath)

        if match_count == DATES_PER_LOCATION:
            break

    if not match_count:
        print("No matching data found")


if __name__ == "__main__":
    main()
