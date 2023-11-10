"""
STAC Data Processing Script

This Python script processes Sentinel-2, Sentinel-1, and DEM (Digital Elevation Model) data. It utilizes the Planetary Computer API for data retrieval and manipulation.

Constants:
- STAC_API: Planetary Computer API endpoint
- S2_BANDS: Bands used in Sentinel-2 data processing

Functions:
- random_date(start_year, end_year): Generate a random date within a specified range.
- get_week(year, month, day): Get the week range for a given date.
- get_conditions(year1, year2): Get random conditions (date, year, month, day, cloud cover) within a specified year range.
- search_sentinel2(week, aoi_gdf): Search for Sentinel-2 items within a given week and area of interest.
- search_sentinel1(BBOX, catalog, week): Search for Sentinel-1 items within a given bounding box, STAC catalog, and week.
- search_sentinel1_calc_max_area(BBOX, catalog, week): Search for Sentinel-1 items within a given bounding box, STAC catalog, and week, selecting the scene with the maximum overlap.
- search_dem(BBOX, catalog, week, s2_items): Search for DEM items within a given bounding box, STAC catalog, week, and Sentinel-2 items.
- make_dataarrays(s2_items, s1_items, dem_items, BBOX, resolution, epsg): Create xarray DataArrays for Sentinel-2, Sentinel-1, and DEM data.
- merge_datarrays(da_sen2, da_sen1, da_dem): Merge xarray DataArrays for Sentinel-2, Sentinel-1, and DEM.
- process(year1, year2, aoi, resolution, epsg): Process Sentinel-2, Sentinel-1, and DEM data for a specified time range, area of interest, resolution, and EPSG code.
"""

import random
from datetime import datetime, timedelta
import geopandas as gpd
import pystac
import pystac_client
import planetary_computer as pc
import stackstac
import xarray as xr
import numpy as np
from shapely.geometry import box, shape, Point
import leafmap
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
from getpass import getpass
import os
import json

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]


def random_date(start_year, end_year):
    """
    Generate a random date within the specified range.

    Parameters:
    - start_year (int): The starting year of the date range.
    - end_year (int): The ending year of the date range.

    Returns:
    - datetime: A randomly generated date within the specified range.
    """
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    random_date = start_date + timedelta(days=random_days)
    return random_date


def get_week(year, month, day):
    """
    Get the week range (start_date/end_date) for a given date.

    Parameters:
    - year (int): The year of the date.
    - month (int): The month of the date.
    - day (int): The day of the date.

    Returns:
    - str: A string representing the start and end dates of the week in the format 'start_date/end_date'.
    """
    date = datetime(year, month, day)
    start_of_week = date - timedelta(days=date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    start_date_str = start_of_week.strftime('%Y-%m-%d')
    end_date_str = end_of_week.strftime('%Y-%m-%d')
    return f"{start_date_str}/{end_date_str}"


def get_conditions(year1, year2):
    """
    Get random conditions (date, year, month, day, cloud cover) within the specified year range.

    Parameters:
    - year1 (int): The starting year of the date range.
    - year2 (int): The ending year of the date range.

    Returns:
    - tuple: A tuple containing date, year, month, day, and a constant cloud cover value.
    """
    date = random_date(year1, year2)
    YEAR = date.year
    MONTH = date.month
    DAY = date.day
    CLOUD = 50
    return date, YEAR, MONTH, DAY, CLOUD


def search_sentinel2(week, aoi):
    """
    Search for Sentinel-2 items within a given week and area of interest (AOI).

    Parameters:
    - week (str): The week in the format 'start_date/end_date'.
    - aoi (shapely.geometry.base.BaseGeometry): Geometry object for an Area of Interest (AOI).

    Returns:
    - tuple: A tuple containing the STAC catalog, Sentinel-2 items, Sentinel-2 GeoDataFrame, and the bounding box (BBOX).
    """

    CENTROID = aoi.centroid
    BBOX = aoi.bounds

    geom_CENTROID = Point(CENTROID.x, CENTROID.y)  # Create point geom object from centroid

    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)

    search: pystac_client.item_search.ItemSearch = catalog.search(
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {
                    "op": "s_intersects",
                    "args": [{"property": "geometry"}, geom_CENTROID.__geo_interface__],
                },
                {"op": "anyinteracts", "args": [{"property": "datetime"}, week]},
                {"op": "=", "args": [{"property": "collection"}, "sentinel-2-l2a"]},
                {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 50]},
                {"op": "<=", "args": [{"property": "s2:nodata_pixel_percentage"}, 20]},
            ],
        },
    )

    s2_items = search.get_all_items()
    print(f"Found {len(s2_items)} Sentinel-2 items")

    s2_items_gdf = gpd.GeoDataFrame.from_features(s2_items.to_dict())

    BBOX = s2_items_gdf.iloc[0].geometry.bounds

    return catalog, s2_items, s2_items_gdf, BBOX


def search_sentinel1(BBOX, catalog, week):
    """
    Search for Sentinel-1 items within a given bounding box (BBOX), STAC catalog, and week.

    Parameters:
    - BBOX (tuple): Bounding box coordinates in the format (minx, miny, maxx, maxy).
    - catalog (pystac.Catalog): STAC catalog containing Sentinel-1 items.
    - week (str): The week in the format 'start_date/end_date'.

    Returns:
    - tuple: A tuple containing Sentinel-1 items and Sentinel-1 GeoDataFrame.
    """

    geom_BBOX = box(*BBOX)  # Create poly geom object from the bbox

    search: pystac_client.item_search.ItemSearch = catalog.search(
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {
                    "op": "s_intersects",
                    "args": [{"property": "geometry"}, geom_BBOX.__geo_interface__],
                },
                {"op": "anyinteracts", "args": [{"property": "datetime"}, week]},
                {"op": "<=", "args": [{"property": "sat:orbit_state"}, "descending"]},
                {"op": "=", "args": [{"property": "collection"}, "sentinel-1-rtc"]},
            ],
        },
    )
    s1_items = search.item_collection()
    print(f"Found {len(s1_items)} Sentinel-1 items")

    s1_gdf = gpd.GeoDataFrame.from_features(s1_items.to_dict())
    # TODO: Find a way to get minimal number of S1 tiles to cover the entire S2 bbox
    s1_gdf["overlap"] = s1_gdf.intersection(box(*BBOX)).area
    s1_gdf.sort_values(by="overlap", inplace=True)

    return s1_items, s1_gdf


def search_sentinel1_calc_max_area(BBOX, catalog, week):
    """
    Search for Sentinel-1 items within a given bounding box (BBOX), STAC catalog, and week.

    Parameters:
    - BBOX (tuple): Bounding box coordinates in the format (minx, miny, maxx, maxy).
    - catalog (pystac.Catalog): STAC catalog containing Sentinel-1 items.
    - week (str): The week in the format 'start_date/end_date'.

    Returns:
    - tuple: A tuple containing the selected Sentinel-1 item and Sentinel-1 GeoDataFrame.
    """

    geom_BBOX = box(*BBOX)  # Create poly geom object from the bbox

    search: pystac_client.item_search.ItemSearch = catalog.search(
        filter_lang="cql2-json",
        filter={
            "op": "and",
            "args": [
                {
                    "op": "s_intersects",
                    "args": [{"property": "geometry"}, geom_BBOX.__geo_interface__],
                },
                {"op": "anyinteracts", "args": [{"property": "datetime"}, week]},
                {"op": "=", "args": [{"property": "collection"}, "sentinel-1-rtc"]},
            ],
        },
    )
    s1_items = search.get_all_items()
    print(f"Found {len(s1_items)} Sentinel-1 items")

    s1_gdf = gpd.GeoDataFrame.from_features(s1_items.to_dict())
    s1_gdf["overlap"] = s1_gdf.intersection(box(*BBOX)).area

    # Choose the scene with the maximum overlap
    selected_scene = s1_gdf.loc[s1_gdf["overlap"].idxmax()]
    print(selected_scene)
    selected_item_id = selected_scene["id"]
    selected_item = catalog.get_item(selected_item_id)

    return selected_item, s1_gdf


def search_dem(BBOX, catalog, week, s2_items):
    """
    Search for Digital Elevation Model (DEM) items within a given bounding box (BBOX), STAC catalog, week, and Sentinel-2 items.

    Parameters:
    - BBOX (tuple): Bounding box coordinates in the format (minx, miny, maxx, maxy).
    - catalog (pystac.Catalog): STAC catalog containing DEM items.
    - week (str): The week in the format 'start_date/end_date'.
    - s2_items (list): List of Sentinel-2 items.

    Returns:
    - tuple: A tuple containing DEM items and DEM GeoDataFrame.
    """
    search = catalog.search(
        collections=["cop-dem-glo-30"], bbox=BBOX
    )
    dem_items = search.item_collection()
    print(f"Found {len(dem_items)} items")

    dem_gdf = gpd.GeoDataFrame.from_features(dem_items.to_dict())

    dem_gdf.set_crs(epsg=4326, inplace=True)
    dem_gdf = dem_gdf.to_crs(epsg=s2_items[0].properties["proj:epsg"])
    return dem_items, dem_gdf


def make_dataarrays(s2_items, s1_items, dem_items, BBOX, resolution, epsg):
    """
    Create xarray DataArrays for Sentinel-2, Sentinel-1, and DEM data.

    Parameters:
    - s2_items (list): List of Sentinel-2 items.
    - s1_items (list): List of Sentinel-1 items.
    - dem_items (list): List of DEM items.
    - BBOX (tuple): Bounding box coordinates in the format (minx, miny, maxx, maxy).
    - resolution (int): Spatial resolution.
    - epsg (int): EPSG code for the coordinate reference system.

    Returns:
    - tuple: A tuple containing xarray DataArrays for Sentinel-2, Sentinel-1, and DEM.
    """
    da_sen2: xr.DataArray = stackstac.stack(
        items=s2_items[0],
        epsg=26910,  # UTM Zone 10N
        assets=S2_BANDS,
        bounds_latlon=BBOX,  # W, S, E, N
        resolution=10,  # Spatial resolution of 10 metres
        xy_coords="center",  # pixel centroid coords instead of topleft corner
        dtype=np.float32,
        fill_value=np.nan,
    )

    da_sen1: xr.DataArray = stackstac.stack(
        items=s1_items[1:],  # To only accept the same orbit state and date. Need better way to do this.
        assets=["vh", "vv"],  # SAR polarizations
        epsg=26910,  # UTM Zone 10N
        bounds_latlon=BBOX,  # W, S, E, N
        xy_coords="center",  # pixel centroid coords instead of topleft corner
        dtype=np.float32,
        fill_value=np.nan,
    )

    # To fix TypeError: Invalid value for attr 'spec'
    da_sen1.attrs["spec"] = str(da_sen1.spec)

    # To fix ValueError: unable to infer dtype on variable None
    for key, val in da_sen1.coords.items():
        if val.dtype == "object":
            print("Deleting", key)
            da_sen1 = da_sen1.drop_vars(names=key)

    # Create xarray.Dataset datacube with VH and VV channels from SAR
    da_vh: xr.DataArray = da_sen1.sel(band="vh", drop=True).rename("vh")
    da_vv: xr.DataArray = da_sen1.sel(band="vv", drop=True).rename("vv")
    ds_sen1: xr.Dataset = xr.merge(objects=[da_vh, da_vv], join="override")

    # print(ds_sen1)

    # da_sen1 = da_sen1.drop([da_sen1.time[0].values], dim='time') # Remove first scene which has ascending orbit

    da_dem: xr.DataArray = stackstac.stack(
        items=dem_items,
        epsg=26910,  # UTM Zone 10N
        bounds_latlon=BBOX,  # W, S, E, N
        resolution=10,  # Spatial resolution of 10 metres
        xy_coords="center",  # pixel centroid coords instead of topleft corner
        dtype=np.float32,
        fill_value=np.nan,
    )

    _, index = np.unique(da_dem['time'], return_index=True)  # Remove redundant time
    da_dem = da_dem.isel(time=index)

    return da_sen2, da_sen1, da_dem


def merge_datarrays(da_sen2, da_sen1, da_dem):
    """
    Merge xarray DataArrays for Sentinel-2, Sentinel-1, and DEM.

    Parameters:
    - da_sen2 (xr.DataArray): xarray DataArray for Sentinel-2 data.
    - da_sen1 (xr.DataArray): xarray DataArray for Sentinel-1 data.
    - da_dem (xr.DataArray): xarray DataArray for DEM data.

    Returns:
    - xr.DataArray: Merged xarray DataArray.
    """
    da_merge = xr.merge([da_sen2, da_sen1, da_dem], compat='override')
    print(da_merge)
    return da_merge


def process(year1, year2, aoi, resolution, epsg):
    """
    Process Sentinel-2, Sentinel-1, and DEM data for a specified time range, area of interest (AOI),
    resolution, and EPSG code.

    Parameters:
    - year1 (int): The starting year of the date range.
    - year2 (int): The ending year of the date range.
    - aoi (shapely.geometry.base.BaseGeometry): Geometry object for an Area of Interest (AOI).
    - resolution (int): Spatial resolution.
    - epsg (int): EPSG code for the coordinate reference system.

    Returns:
    - xr.DataArray: Merged xarray DataArray containing processed data.
    """

    date, YEAR, MONTH, DAY, CLOUD = get_conditions(year1, year2)
    week = get_week(YEAR, MONTH, DAY)

    catalog, s2_items, s2_items_gdf, BBOX = search_sentinel2(week, aoi)

    s1_items, s1_gdf = search_sentinel1(BBOX, catalog, week)
    # s1_items, s1_gdf = search_sentinel1_calc_max_area(BBOX, catalog, week) # WIP


    dem_items, dem_gdf = search_dem(BBOX, catalog, week, s2_items)

    da_sen2, da_sen1, da_dem = make_dataarrays(s2_items, s1_items, dem_items, BBOX, resolution, epsg)

    da_merge = merge_datarrays(da_sen2, da_sen1, da_dem)
    return da_merge

# EXAMPLE
california_tile = gpd.read_file("ca.geojson") 
sample = california_tile.sample(1)
aoi = sample.iloc[0].geometry
process(2017, 2023,  aoi, 10, 26910)