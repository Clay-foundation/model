from pathlib import Path

from pystac import Item

wd = Path("/home/tam/Desktop/clay-v1-data/items")

# Change landsat
for item_path in wd.glob("landsat*.json"):
    item = Item.from_file(item_path)
    for key, asset in item.assets.items():
        # Make S3 url main source
        if key in ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]:
            asset.href = asset.extra_fields["alternate"]["s3"]["href"]

    item.save_object()
