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
- make_datasets(s2_items, s1_items, dem_items, resolution):
      Create xarray Datasets for Sentinel-2, Sentinel-1, and DEM data.
- process(aoi, start_year, end_year, resolution, cloud_cover_percentage, nodata_pixel_percentage):
      Process Sentinel-2, Sentinel-1, and DEM data for a specified time range,
      area of interest, and resolution.
"""

import random
from datetime import timedelta

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import stackstac
import xarray as xr
from pystac import ItemCollection
from shapely.geometry import box

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
SPATIAL_RESOLUTION = 10
CLOUD_COVER_PERCENTAGE = 50
NODATA_PIXEL_PERCENTAGE = 20


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


def search_sentinel2(date_range, aoi, cloud_cover_percentage, nodata_pixel_percentage):
    """
    Search for Sentinel-2 items within a given date range and area of interest (AOI)
    with specified conditions.

    Parameters:
    - date_range (str): The date range in the format 'start_date/end_date'.
    - aoi (shapely.geometry.base.BaseGeometry): Geometry object for an Area of
        Interest (AOI).
    - cloud_cover_percentage (int): Maximum acceptable cloud cover percentage
        for Sentinel-2 images.
    - nodata_pixel_percentage (int): Maximum acceptable percentage of nodata
        pixels in Sentinel-2 images.

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
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)

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

    least_clouds = s2_items_gdf.sort_values(
        by=["eo:cloud_cover"], ascending=True
    ).index[0]

    s2_items_gdf = s2_items_gdf.iloc[least_clouds]

    # Get the datetime for the filtered Sentinel 2 dataframe
    # containing the least nodata and least cloudy scene
    s2_items_gdf_datetime = s2_items_gdf["datetime"]
    for item in s2_items:
        if item.properties["datetime"] == s2_items_gdf_datetime:
            s2_item = item
        else:
            continue

    bbox = s2_items_gdf.iloc[0].bounds

    epsg = s2_item.properties["proj:epsg"]
    print("EPSG code based on Sentinel-2 item: ", epsg)

    return catalog, s2_item, bbox


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

    if s1_items is None:
        return False
    
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
            elif orbit == most_overlap_orbit and intersection.covers(geom_bbox) == False:
                intersection = row.geometry
                selected_item_ids.append(row.id)
                intersection = intersection.intersection(row.geometry)
            elif orbit == most_overlap_orbit and intersection.covers(geom_bbox) == True:
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

    # Create xarray.Dataset datacube with all 10m and 20m bands from Sentinel-2
    da_s2_0: xr.DataArray = da_sen2.sel(band="B02", drop=True).rename("B02").squeeze()
    da_s2_1: xr.DataArray = da_sen2.sel(band="B03", drop=True).rename("B03").squeeze()
    da_s2_2: xr.DataArray = da_sen2.sel(band="B04", drop=True).rename("B04").squeeze()
    da_s2_3: xr.DataArray = da_sen2.sel(band="B05", drop=True).rename("B05").squeeze()
    da_s2_4: xr.DataArray = da_sen2.sel(band="B06", drop=True).rename("B06").squeeze()
    da_s2_5: xr.DataArray = da_sen2.sel(band="B07", drop=True).rename("B07").squeeze()
    da_s2_6: xr.DataArray = da_sen2.sel(band="B08", drop=True).rename("B08").squeeze()
    da_s2_7: xr.DataArray = da_sen2.sel(band="B8A", drop=True).rename("B8A").squeeze()
    da_s2_8: xr.DataArray = da_sen2.sel(band="B11", drop=True).rename("B11").squeeze()
    da_s2_9: xr.DataArray = da_sen2.sel(band="B11", drop=True).rename("B11").squeeze()
    da_s2_10: xr.DataArray = da_sen2.sel(band="SCL", drop=True).rename("SCL").squeeze()

    ds_sen2: xr.Dataset = xr.merge(
        objects=[
            da_s2_0,
            da_s2_1,
            da_s2_2,
            da_s2_3,
            da_s2_4,
            da_s2_5,
            da_s2_6,
            da_s2_7,
            da_s2_8,
            da_s2_9,
            da_s2_10,
        ],
        join="override",
    )
    ds_sen2.assign(time=da_sen2.time)

    da_sen1: xr.DataArray = stackstac.stack(
        items=s1_items,
        assets=["vh", "vv"],
        epsg=int(da_sen2.epsg),
        bounds=da_sen2.spec.bounds,
        resolution=resolution,
        dtype=np.float32,
        fill_value=np.nan,
    )

    # Create xarray.Dataset datacube with VH and VV channels from SAR
    da_sen1 = stackstac.mosaic(da_sen1, dim="time")
    da_vh: xr.DataArray = da_sen1.sel(band="vh", drop=True).squeeze().rename("vh")
    da_vv: xr.DataArray = da_sen1.sel(band="vv", drop=True).squeeze().rename("vv")
    ds_sen1: xr.Dataset = xr.merge(objects=[da_vh, da_vv], join="override")

    da_dem: xr.DataArray = stackstac.stack(
        items=dem_items,
        epsg=int(da_sen2.epsg),
        bounds=da_sen2.spec.bounds,
        resolution=resolution,
        dtype=np.float32,
        fill_value=np.nan,
    )

    da_dem: xr.DataArray = stackstac.mosaic(da_dem, dim="time").squeeze().rename("DEM")

    return ds_sen2, ds_sen1, da_dem


def process(aoi, start_year, end_year, resolution, cloud_cover_percentage, nodata_pixel_percentage):
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
    year = random.randint(start_year, end_year)
    date_range = f"{year}-01-01/{year}-12-31"
    catalog, s2_items, bbox = search_sentinel2(
        date_range, aoi, cloud_cover_percentage, nodata_pixel_percentage
    )

    surrounding_days = get_surrounding_days(s2_items.datetime, interval_days=3)
    print("Searching S1 in date range", surrounding_days)

    s1_items = search_sentinel1(bbox, catalog, surrounding_days)

    if s1_items == False:
        catalog, s2_items, bbox = search_sentinel2(
                date_range, aoi, cloud_cover_percentage, nodata_pixel_percentage
                )
        
        surrounding_days = get_surrounding_days(s2_items.datetime, interval_days=3)
        print("Searching S1 in date range", surrounding_days)
        s1_items = search_sentinel1(bbox, catalog, surrounding_days)
    

    dem_items = search_dem(bbox, catalog)

    ds_sen2, ds_sen1, da_dem = make_datasets(
        s2_items,
        s1_items,
        dem_items,
        resolution,
    )

    ds_merge = xr.merge([ds_sen2, ds_sen1, da_dem], compat="override")

    return ds_merge


def main():
    tiles = gpd.read_file("scripts/data/mgrs_sample.geojson")
    sample = tiles.sample(1, random_state=45)
    aoi = sample.iloc[0].geometry
    start_year = 2017
    end_year = 2023

    merged = process(
        aoi, start_year, end_year, SPATIAL_RESOLUTION, CLOUD_COVER_PERCENTAGE, NODATA_PIXEL_PERCENTAGE
    )
    return merged

# main()