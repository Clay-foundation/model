#!/usr/bin/env python3
"""
This script contains functions to process Sentinel-2, Sentinel-1 and DEM data based on the
Cloud to Street - Microsoft Flood benchmark dataset
(see: https://beta.source.coop/repositories/c2sms/c2smsfloods/description/).
It produces datacubes that reflect the structure of training data for a pretext MAE model.
"""
import io
from datetime import datetime, timedelta

import boto3
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

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
S2_BANDS_c2smsfloods = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS_c2smsfloods = ["VV", "VH"]

BUCKET_NAME = "clay-benchmark"
PREFIX = "c2smsfloods/chips/"

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


def make_datasets(s2_items, s1_items, dem_items, resolution, bbox):
    """
    Creates datasets by stacking Sentinel-2, Sentinel-1, and DEM data arrays.

    Args:
    - s2_items (list): Sentinel-2 items obtained from the STAC API.
    - s1_items (list): Sentinel-1 items obtained from the STAC API.
    - dem_items (list): DEM (Digital Elevation Model) items obtained from the STAC API.
    - resolution (int): Resolution for processing the datasets.
    - bbox (list): Bounding box coordinates.

    Returns:
    - xarray.DataArray: A concatenated xarray DataArray containing stacked Sentinel-2, Sentinel-1, and DEM data.
    """
    da_sen2: xr.DataArray = stackstac.stack(
        items=s2_items,
        assets=S2_BANDS,
        resolution=resolution,
        bounds=bbox,
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
    bbox,
    bbox_4326,
    resolution,
):
    """
    Processes Sentinel-2, Sentinel-1 and DEM items retrieved from STAC API to create datasets.

    Args:
    - s2_item (pystac.Item): Sentinel-2 item obtained from the STAC API.
    - bbox (list): Bounding box coordinates in the original CRS format.
    - bbox_4326 (list): Bounding box coordinates in EPSG:4326 format.
    - resolution (int): Resolution for processing the datasets.

    Returns:
    - xarray.DataArray or None: A concatenated xarray DataArray containing stacked Sentinel-2,
        Sentinel-1, and DEM data if successful, else None.
    """
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)

    for i in range(S1_MATCH_ATTEMPTS):
        print("Sentinel-2 item: ", s2_item)
        surrounding_days = get_surrounding_days(s2_item.datetime, interval_days=3)
        print("Searching S1 in date range", surrounding_days)

        s1_items = search_sentinel1(bbox_4326, catalog, surrounding_days)
        if s1_items:
            S1_granules_modified = [
                f"{'_'.join(i.split('_')[:-2])}_rtc" for i in S1_granules
            ]
            s1_items = [i for i in s1_items if i.id in S1_granules_modified]
            print("Sentinel-1 items matching benchmark: ", s1_items)
            break
    # if i == S1_MATCH_ATTEMPTS - 1:
    #    raise ValueError(
    #        f"No match for S1 scenes found after {S1_MATCH_ATTEMPTS} attempts."
    #    )
    if s1_items:
        print(f"We retrieved {len(s1_items)} S1 items.")
        dem_items = search_dem(bbox_4326, catalog)

        date = s2_item.properties["datetime"][:10]
        result = make_datasets(s2_item, s1_items, dem_items, resolution, bbox)

        return result, s1_items
    else:
        return None


def find_stac_item_by_granule_name(
    catalog, collection_name, granule_name, bbox, time_range
):
    """
    Search for a STAC item within a specific collection by granule name, bounding box, and time range.

    Args:
    - catalog (pystac.Catalog): The PySTAC catalog containing the collection.
    - collection_name (str): The name of the collection within the catalog to search.
    - granule_name (str or None): The granule name to find within the collection. If None, returns the first item.
    - bbox (list or tuple): Bounding box coordinates in [minx, miny, maxx, maxy] order.
    - time_range (str): Time range in ISO8601 format, e.g., '2023-01-01T00:00:00Z/2023-12-31T23:59:59Z'.

    Returns:
    - pystac.Item or None: The STAC item corresponding to the provided granule name within the specified collection.
      Returns None if the granule name is not found or if no granule name is provided (granule_name=None).
    """
    search = catalog.search(
        collections=[collection_name], bbox=bbox, datetime=f"{time_range}/{time_range}"
    )
    items = search.item_collection()
    print(f"Found {len(items)} items")

    for item in items:
        if granule_name and granule_name in item.id:
            break
        elif granule_name is None:
            break
    if items:
        return items[0]
    else:
        return None


def list_objects_recursive(client, bucket_name, prefix=""):
    """
    List all objects (file keys) in an S3 bucket recursively under a specified prefix.

    Args:
    - client (boto3.client): An initialized Boto3 S3 client.
    - bucket_name (str): The name of the S3 bucket.
    - prefix (str): The prefix (directory path) within the bucket to search for objects (optional).

    Returns:
    - list: A list of file keys (object keys) found under the specified prefix.
    """
    paginator = client.get_paginator("list_objects_v2")

    # Initial page request with the prefix
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    file_keys = []
    for page in page_iterator:
        if "Contents" in page:
            # Append the file keys (object keys) from the current page
            file_keys.extend([obj["Key"] for obj in page["Contents"]])

    # Return the list of file keys
    return file_keys


def get_image_granules(bucket_name, prefix):
    """
    Get granules of Sentinel-2 (S2) and Sentinel-1 (S1) images from an S3 bucket.

    Args:
    - bucket_name (str): The name of the S3 bucket.
    - prefix (str): The prefix (directory path) in the S3 bucket.

    Returns:
    - tuple: A tuple containing lists of S2 images, S1 images, S2 granules, S1 granules and tile IDs.
    """
    # Initialize Boto3 S3 client
    s3 = boto3.client("s3")

    # List objects in the specified prefix (directory) in the bucket
    files_in_s3 = list_objects_recursive(s3, bucket_name, prefix)

    # Filter S2 and S1 images
    S2_images = [i for i in files_in_s3 if "/s2/" in i]
    S1_images = [i for i in files_in_s3 if "/s1/" in i]
    S2_images = list(set(["/".join(i.split("/")[:-1]) for i in S2_images]))
    S1_images = list(set(["/".join(i.split("/")[:-1]) for i in S1_images]))

    # Extract granules from image paths
    S2_granules = list(set([i.split("/")[-1] for i in S2_images]))
    S1_granules = list(set([i.split("/")[-1] for i in S1_images]))

    tile_ids = [i.split("/")[2] for i in S2_images]

    return S2_images, S1_images, S2_granules, S1_granules, tile_ids


def process_image_granule(image_granule_path, bucket_name, bands=None):
    """
    Extracts information from an image granule stored in an AWS S3 bucket.

    Args:
    - image_granule_path (str): The path to the image granule in the S3 bucket.
    - bucket_name (str): The name of the AWS S3 bucket containing the image granule.
    - bands (list, optional): List of bands to consider from the image. Defaults to None.

    Returns:
    - tuple: A tuple containing:
        - granule_name (str): The name of the processed granule.
        - time_stamp (str): The timestamp extracted from the granule's name.
        - bounds (list): The bounds of the processed granule.
        - bounds_4326 (list): The bounds of the processed granule in EPSG:4326.
        - crs: The CRS (Coordinate Reference System) of the granule.
    """
    # S3 bucket and file information
    b = bands[0]  # Take first band just for metadata
    print("image_granule_path: ", image_granule_path)
    file_key = f"{image_granule_path}/{b}.tif"  # File's key in S3 bucket

    # Initialize a Boto3 S3 client
    s3 = boto3.client("s3")

    # Load the image file from S3 directly into memory using rasterio
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    with rasterio.io.MemoryFile(obj["Body"].read()) as memfile:
        with memfile.open() as dataset:
            data_array = rioxarray.open_rasterio(dataset)

    data_array = data_array.rename(f"{b}")
    print("CRS: ", data_array.rio.crs)
    # Get bounds
    bbox = data_array.rio.bounds()
    gbbox = gpd.GeoDataFrame(geometry=[box(bbox[0], bbox[1], bbox[2], bbox[3])])
    gbbox.crs = data_array.rio.crs
    gbbox_4326 = gbbox.to_crs("EPSG:4326")
    granule = image_granule_path.split("/")[-1]
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

    return (
        granule_name,
        time_stamp,
        gbbox.total_bounds.tolist(),
        gbbox_4326.total_bounds.tolist(),
        gbbox.crs,
    )


def write_stack_to_s3(merged, bucket_name, granule_name, s1_granule_name, crs, tile_id):
    """
    Writes an xarray dataset stack to a GeoTIFF in an AWS S3 bucket.

    Args:
    - merged (xarray.Dataset): The dataset to be written to a GeoTIFF.
    - bucket_name (str): The name of the AWS S3 bucket to write the GeoTIFF file.
    - granule_name (str): The name of the granule.
    - crs (str): The CRS (Coordinate Reference System) information.
    - tile_id (str): The ID of the tile.

    Returns:
    - None: The GeoTIFF file is written to the specified S3 bucket and path.
    """
    merged = merged.drop_sel(band="SCL")
    # Write tile to tempdir
    name = f"{granule_name}_{s1_granule_name}.tif"

    # Set the geospatial information
    merged.rio.set_spatial_dims("x", "y", inplace=True)
    merged.rio.write_crs(crs, inplace=True)
    merged.attrs["long_name"] = [str(x.values) for x in merged.band]

    # Write the dataset to a GeoTIFF file in memory
    tif_bytes = io.BytesIO()
    merged.rio.to_raster(tif_bytes, driver="GTiff")

    # Write the GeoTIFF file to S3
    s3 = boto3.client("s3")
    s3.put_object(
        Body=tif_bytes.getvalue(),
        Bucket=bucket_name,
        Key=f"c2smsfloods/datacube/chips_512_v1/{tile_id}/{name}",
    )


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
    collection = "sentinel-2-l2a"
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)
    tile_id = image_granule_path.split("/")[2]
    granule_name, time_stamp, bbox, bbox_4326, crs = process_image_granule(
        image_granule_path, bucket_name=bucket_name, bands=S2_BANDS_c2smsfloods
    )
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)
    S2_L2A_item = find_stac_item_by_granule_name(
        catalog, collection, granule_name, bbox_4326, time_stamp
    )
    S1_items = search_sentinel1(
        bbox_4326, catalog, get_surrounding_days(S2_L2A_item.datetime, interval_days=3)
    )
    S1_granules_modified = [f"{'_'.join(i.split('_')[:-2])}_rtc" for i in S1_granules]
    if S1_items:
        S1_item_passing = [i for i in S1_items if i.id in S1_granules_modified]
        if S2_L2A_item and S1_item_passing:
            merged, s1_items = process(
                S2_L2A_item,
                bbox,
                bbox_4326,
                SPATIAL_RESOLUTION,
            )
            if merged is not None and s1_items is not None:
                merged = merged.compute()
                # print(merged)
                print("s1_items: ", s1_items)
                s1_granule_name = [i.id for i in s1_items]
                write_stack_to_s3(
                    merged, bucket_name, granule_name, s1_granule_name[0], crs, tile_id
                )
                print("written")
            return merged
    else:
        pass


# List objects in the specified prefix (directory) in the bucket
S2_images, S1_images, S2_granules, S1_granules, tile_ids = get_image_granules(
    bucket_name=BUCKET_NAME, prefix=PREFIX
)
print(f"We have {len(S2_images)} images to process.")

for i, c in zip(S2_images, range(len(S2_images))):
    print("Image count: ", c)
    print(f"Processing {i}")
    main(str(i), BUCKET_NAME, PREFIX)
