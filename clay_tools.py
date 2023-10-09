import math
from typing import Any, Dict
from pystac.item import Item
from rasterio.transform import from_origin
import rasterio
from shapely.geometry import shape
import os


def tile_bbox(x: int, y: int,zoom: int, ):
    #Given a [x,y,z] tile, return the bounding box of the tile in [w,s,e,n] format
    return [ tile_lon(x + 0, zoom), tile_lat(y + 1, zoom), tile_lon(x + 1, zoom), tile_lat(y + 0, zoom)]

def tile_lon(x: int, z: int) -> float:
    #Given a [x,y,z] tile, return the longitude of the southwest corner of the tile in degrees
    return x / math.pow(2.0, z) * 360.0 - 180

def tile_lat(y: int, z: int) -> float:
    #Given a [x,y,z] tile, return the latitude of the southwest corner of the tile in degrees
    return math.degrees( math.atan(math.sinh(math.pi - (2.0 * math.pi * y) / math.pow(2.0, z))))

def bbox_to_aoi(bbox):
    #Given a bbox in [w,s,e,n] format, return aoi in [type, coordinates] format
    xmin, ymin, xmax, ymax = bbox
    aoi = {"type": "Polygon", "coordinates": [[[xmin, ymin],[xmin, ymax],[xmax, ymax],[xmax, ymin],[xmin, ymin]]]}
    return aoi

def center_lonlat_of_bbox(bbox):
    #Given a bbox in [w,s,e,n] format, return the center lonlat in [lon, lat] format
    xmin, ymin, xmax, ymax = bbox
    return (xmin + xmax) / 2, (ymin + ymax) / 2

def intersection_percent(item: Item, aoi: Dict[str, Any]) -> float:
    #Given an item and an aoi, return the percent of the item that is in the aoi
    geom_item = shape(item.geometry)
    geom_aoi = shape(aoi)
    intersected_geom = geom_aoi.intersection(geom_item)
    return (intersected_geom.area * 100) / geom_aoi.area

def save_tif_tile(x,y,z, tif_data,folder="./"):
    print("TIF data shape:", tif_data.shape)
    print("TIF data dtype:", tif_data.dtype)
    #create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f"{z}-{x}-{y}.tif"
    bbox = tile_bbox(x, y, z)
    transform = from_origin(bbox[0], bbox[3], (bbox[2]-bbox[0])/tif_data.shape[2], (bbox[3]-bbox[1])/tif_data.shape[1])
    with rasterio.open(folder+"/"+filename, 'w', driver='GTiff', height=tif_data.shape[1], 
                       width=tif_data.shape[2], count=tif_data.shape[0], dtype=str(tif_data.dtype),
                       crs='+proj=latlong', transform=transform) as dst:
        dst.write(tif_data)