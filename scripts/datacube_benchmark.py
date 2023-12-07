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
- process(aoi, start_year, end_year, resolution, cloud_cover_percentage,
          nodata_pixel_percentage):
      Process Sentinel-2, Sentinel-1, and DEM data for a specified time range,
      area of interest, and resolution.
"""
import random
from datetime import datetime, timedelta
import boto3
import click
import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
import rioxarray
import stackstac
import xarray as xr
from pystac import ItemCollection
from shapely.geometry import box
from tile import tiler

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
S2_BANDS_c2smsfloods = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS_c2smsfloods = ["VV", "VH"]

BUCKET_NAME = 'clay-benchmark'
PREFIX = 'c2smsfloods/chips/'

SPATIAL_RESOLUTION = 10
CLOUD_COVER_PERCENTAGE = 50
NODATA_PIXEL_PERCENTAGE = 20
NODATA = 0
S1_MATCH_ATTEMPTS = 10


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
        dtype=np.float32,
        fill_value=np.nan,
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

    result = xr.concat([da_sen2, da_sen1, da_dem], dim="band")
    result = result.rename("tile")
    return result


def process(
    s2_item,
    aoi,
    start_year,
    end_year,
    resolution,
    cloud_cover_percentage,
    nodata_pixel_percentage,
):
    """
    Process Sentinel-2, Sentinel-1, and Copernicus DEM data for a specified
    time range, area of interest (AOI), resolution, EPSG code, cloud cover
    percentage, and nodata pixel percentage.

    Parameters:
    - aoi (shapely.geometry.base.BaseGeometry): Geometry object for an Area of
        Interest (AOI).
    - start_year (int): The starting year of the date range.
    - end_year (int): The ending year of the date range.
    - resolution (int): Spatial resolution.
    - cloud_cover_percentage (int): Maximum acceptable cloud cover percentage
        for Sentinel-2 images.
    - nodata_pixel_percentage (int): Maximum acceptable percentage of nodata
        pixels in Sentinel-2 images.

    Returns:
    - xr.Dataset: Merged xarray Dataset containing processed data.
    """
    #year = random.randint(start_year, end_year)
    date_range = f"{start_year}/{end_year}"
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)

    for i in range(S1_MATCH_ATTEMPTS):
        s2_item, bbox = s2_item, aoi
        print("s2_item: ", s2_item)
        surrounding_days = get_surrounding_days(s2_item.datetime, interval_days=3)
        print("Searching S1 in date range", surrounding_days)

        s1_items = search_sentinel1(bbox, catalog, surrounding_days)

        s1_items = [i for i in s1_items if i in S1_granules]

        if s1_items:
            break

    if i == S1_MATCH_ATTEMPTS - 1:
        raise ValueError(
            f"No match for S1 scenes found after {S1_MATCH_ATTEMPTS} attempts."
        )

    dem_items = search_dem(bbox, catalog)

    date = s2_item.properties["datetime"][:10]

    result = make_datasets(
        s2_item,
        s1_items,
        dem_items,
        resolution,
    )

    return date, result

def find_stac_item_by_granule_name(catalog, collection_name, granule_name, bbox, time_range):
    """
    Find a STAC Item by its granule name within a specific collection in a catalog.

    Args:
    - catalog_url (str): URL to the STAC catalog endpoint.
    - collection_name (str): The name of the collection within the catalog.
    - granule_name (str): The granule name to search for.

    Returns:
    - pystac.Item or None: The STAC Item if found, otherwise None.
    """
    found_item = None
    search = catalog.search(collections=[collection_name], bbox=bbox, datetime=f"{time_range}/{time_range}")
    items = search.item_collection()
    print(f"Found {len(items)} items")

    for item in items:
        if granule_name and granule_name in item.id:
            found_item = item
            break
        elif granule_name is None:
            found_item = item
            break
    return items[0]

def list_objects_recursive(client, bucket_name, prefix=''):
    """
    List all objects (file keys) in an S3 bucket recursively under a specified prefix.

    Args:
    - client (boto3.client): An initialized Boto3 S3 client.
    - bucket_name (str): The name of the S3 bucket.
    - prefix (str): The prefix (directory path) within the bucket to search for objects (optional).

    Returns:
    - list: A list of file keys (object keys) found under the specified prefix.
    """
    paginator = client.get_paginator('list_objects_v2')

    # Initial page request with the prefix
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    file_keys = []
    for page in page_iterator:
        if 'Contents' in page:
            # Append the file keys (object keys) from the current page
            file_keys.extend([obj['Key'] for obj in page['Contents']])

    # Return the list of file keys
    return file_keys

def get_image_granules(bucket_name, prefix):
    """
    Get granules of Sentinel-2 (S2) and Sentinel-1 (S1) images from an S3 bucket.

    Args:
    - bucket_name (str): The name of the S3 bucket.
    - prefix (str): The prefix (directory path) in the S3 bucket.

    Returns:
    - tuple: A tuple containing lists of S2 images, S1 images, S2 granules, and S1 granules.
    """
    # Initialize Boto3 S3 client
    s3 = boto3.client('s3')

    # List objects in the specified prefix (directory) in the bucket
    files_in_s3 = list_objects_recursive(s3, bucket_name, prefix)

    # Filter S2 and S1 images
    S2_images = [i for i in files_in_s3 if '/s2/' in i]
    S1_images = [i for i in files_in_s3 if '/s1/' in i]

    # Extract granules from image paths
    S2_granules = list(set([i.split('/')[-2] for i in S2_images]))
    S1_granules = list(set([i.split('/')[-2] for i in S1_images]))

    tile_ids = [i.split('/')[2] for i in S2_images]

    return S2_images, S1_images, S2_granules, S1_granules, tile_ids

def process_image_granule(image_granule_path, bucket_name, bands=None):
    """
    Process an image granule from an S3 bucket.

    Args:
    - image_granule_path (str): Path to the Sentinel-2 image granule in the S3 bucket.
    - bucket_name (str): The name of the S3 bucket.
    - bands (list): List of bands to consider (optional).

    Returns:
    - str: Extracted date from the granule name.
    """

    # S3 bucket and file information
    b = bands[0]  # Take first band just for metadata
    print("image_granule_path: ", image_granule_path)
    file_key = f"{image_granule_path}/{b}.tif"  # File's key in S3 bucket
    print(file_key)

    # Initialize a Boto3 S3 client
    s3 = boto3.client('s3')

    # Load the image file from S3 directly into memory using rasterio
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    with rasterio.io.MemoryFile(obj['Body'].read()) as memfile:
        with memfile.open() as dataset:
            data_array = rioxarray.open_rasterio(dataset)

    data_array = data_array.rename(f"{b}")
    print("CRS: ", data_array.rio.crs)
    print("Transform: ", data_array.rio.transform())

    # Get bounds
    bbox = data_array.rio.bounds()
    gbbox = gpd.GeoDataFrame(geometry=[box(bbox[0], bbox[1], bbox[2], bbox[3])])
    gbbox.crs = data_array.rio.crs
    geobbox = gbbox.to_crs(f"EPSG:4326")
    granule = image_granule_path.split('/')[-1]
    print("Granule: ", granule)
    if "L1C" in granule:
        granule_name = granule.replace("MSIL1C", "L2A")
        # Extracting the date part of the granule name
        date_str = granule_name[8:16]
    else:
        granule_name = granule
        date_str = granule_name[17:25]
    # Converting the extracted string into a datetime object
    date_format = "%Y%m%d"
    time_stamp = datetime.strptime(date_str, date_format)
    time_stamp = str(time_stamp).split()[0]
    print("Extracted Date as datetime (without hours):", time_stamp)

    return granule_name, time_stamp, geobbox.total_bounds.tolist()

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


def main(image_granule_path, bucket_name, prefix):
    start_year = 2017
    end_year = 2023

    s3 = boto3.client('s3')

    collection = "sentinel-2-l2a"
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)
    tile_id = image_granule_path.split('/')[2]
    image_granule_path = '/'.join(image_granule_path.split('/')[:-1])
    granule_name, time_stamp, bbox = process_image_granule(image_granule_path, bucket_name=bucket_name, bands=S2_BANDS_c2smsfloods)
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)
    S2_L2A_item = find_stac_item_by_granule_name(catalog, collection, granule_name, bbox, time_stamp)

    date, merged = process(
        S2_L2A_item,
        bbox,
        time_stamp,
        time_stamp,
        SPATIAL_RESOLUTION,
        CLOUD_COVER_PERCENTAGE,
        NODATA_PIXEL_PERCENTAGE,
    )
    merged = merged.compute()
    print(merged)
    # mgrs = 'na'
    # tiler(merged, date, mgrs)

# List objects in the specified prefix (directory) in the bucket
S2_images, S1_images, S2_granules, S1_granules, tile_ids = get_image_granules(bucket_name=BUCKET_NAME, prefix=PREFIX)  # Include the folder path or prefix
# Test
S2_images = S2_images[0:4]
S1_images = S1_images[0:4]

for i in S2_images:
    main(str(i), BUCKET_NAME, PREFIX)