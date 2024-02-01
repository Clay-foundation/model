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
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
import stackstac
import xarray as xr
from pystac import ItemCollection
from rasterio.enums import Resampling
from shapely import Point
from shapely.geometry import box

STAC_API = "https://earth-search.aws.element84.com/v1"
STAC_API_S2_DEM = "https://planetarycomputer.microsoft.com/api/stac/v1"

BAND_GROUPS = {
    "rgb": ["red", "green", "blue"],
    "rededge": ["rededge1", "rededge2", "rededge3", "nir08"],
    "nir": [
        "nir",
    ],
    "swir": ["swir16", "swir22"],
    "sar": ["vv", "vh"],
}
SPATIAL_RESOLUTION = 10
CLOUD_COVER_PERCENTAGE = 80
NODATA_PIXEL_PERCENTAGE = 20
NODATA = 0
S1_MATCH_ATTEMPTS = 20
DATES_PER_LOCATION = 3

EPSG_AOI = 32642  # Pakistan # 32644 # Nepal # 32638 # Georgia

# lat/lon
POI = 39.99146, -8.06256  # Pedrogao, Portugal
POI = 37.30939, -8.57207  # Monchique Portugal
POI = 41.921592, 41.982836  # Ozurgeti, Georgia
POI = 29.232631, 81.218995  # Achham District, Nepal
POI = 26.776567, 68.287374  # Padidan, Pakistan
# POI = 29.5672, 116.1346  # Poyang, China
# POI = 21.8978476,106.2495839  # Bac Son, Vietnam
# POI = 10.22651, 105.21669  # Mekong delta, Vietnam


# Start date and end date
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 1, 30)


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

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        datetime=f"{date_range}",
        bbox=aoi,
        max_items=100,
        query={"eo:cloud_cover": {"lt": 100}},
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

    return s2_item, aoi


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


def make_datasets(s2_items, s1_items, dem_items, resolution, aoi, bounds):
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
        resolution=resolution,
        bounds=bounds,
        epsg=int(EPSG_AOI),
        snap_bounds=False,
        dtype="float32",
        rescale=False,
        fill_value=0,
        assets=BAND_GROUPS["rgb"] + BAND_GROUPS["nir"] + BAND_GROUPS["swir"],
        resampling=Resampling.bilinear,
    )

    da_sen1: xr.DataArray = stackstac.stack(
        items=s1_items,
        assets=["vh", "vv"],
        epsg=int(EPSG_AOI),
        bounds=bounds,
        resolution=resolution,
        dtype=np.float32,
        fill_value=0,  # np.nan,
    )

    da_dem: xr.DataArray = stackstac.stack(
        items=dem_items,
        epsg=int(EPSG_AOI),
        bounds=bounds,
        resolution=resolution,
        dtype=np.float32,
        fill_value=0,  # np.nan,
    )

    print("da_sen2: ", da_sen2.shape)
    print("da_sen1: ", da_sen1.shape)
    print("da_dem: ", da_dem.shape)

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
    bounds,
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
    catalog_S2_DEM = pystac_client.Client.open(
        STAC_API_S2_DEM, modifier=pc.sign_inplace
    )

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

        s1_items = search_sentinel1(bbox, catalog_S2_DEM, surrounding_days)

        if s1_items:
            break

    if i == S1_MATCH_ATTEMPTS - 1:
        print(
            "No match for S1 scenes found for date range "
            f"{date_range} after {S1_MATCH_ATTEMPTS} attempts."
        )
        return None, None

    dem_items = search_dem(bbox, catalog_S2_DEM)

    date = s2_item.properties["datetime"][:10]

    result = make_datasets(s2_item, s1_items, dem_items, resolution, aoi, bounds)

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


def main(poi, bounds, start, end):
    date_ranges = [f"{start}/{end}"]

    geometry = box(*bounds)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[geometry])
    bbox = (poi[1] - 1e-5, poi[0] - 1e-5, poi[1] + 1e-5, poi[0] + 1e-5)

    match_count = 0
    for date_range in date_ranges:
        print(f"Processing data for date range {date_range}")
        date, pixels = process(
            bbox,
            bounds,
            date_range,
            SPATIAL_RESOLUTION,
            CLOUD_COVER_PERCENTAGE,
            NODATA_PIXEL_PERCENTAGE,
        )

        print("date, pixels: ", date, pixels)
        if date is None:
            continue
        else:
            match_count += 1

        pixels = [part.compute() for part in pixels]
        print(pixels)

        outdir = Path("data/minicubes")
        assert outdir.exists()

        # Write tile to output dir
        for tile in pixels:
            name = "{dir}/claytile_{date}.tif".format(
                dir=outdir,
                date=date.replace("-", ""),
            )
            tile.rio.to_raster(name, compress="deflate")

            with rasterio.open(name, "r+") as rst:
                rst.update_tags(date=date)

        if match_count == DATES_PER_LOCATION:
            break

    if not match_count:
        raise ValueError("No matching data found")


# List to store date ranges
dates = []

# Loop through the dates and create date ranges
while START_DATE < END_DATE:
    end_of_range = START_DATE + timedelta(days=14)
    end_of_range = min(
        end_of_range, END_DATE
    )  # Ensure end_of_range doesn't exceed end_date
    dates.append([START_DATE.strftime("%Y-%m-%d"), end_of_range.strftime("%Y-%m-%d")])
    start_date = end_of_range + timedelta(days=1)

print("List of dates:")
print(dates)
print("LEN DATES: ", len(dates))

poidf = gpd.GeoDataFrame(
    pd.DataFrame(),
    crs="EPSG:4326",
    geometry=[Point(POI[1], POI[0])],
).to_crs(f"EPSG:{EPSG_AOI}")  # 32644 Nepal 32638 # Georgia

coords = poidf.iloc[0].geometry.coords[0]

bounds = (
    coords[0] - 2560,
    coords[1] - 2560,
    coords[0] + 2560,
    coords[1] + 2560,
)

if __name__ == "__main__":
    for date in dates:
        main(POI, bounds, date[0], date[1])
