import warnings
from pathlib import Path

from chip_indexer import LandsatIndexer, NoStatsChipIndexer, Sentinel2Indexer
from pystac import Item

warnings.filterwarnings(
    "ignore",
    message=(
        "The argument 'infer_datetime_format' is deprecated and will"
        " be removed in a future version. A strict version of it is now "
        "the default, see https://pandas.pydata.org/pdeps/0004-consistent"
        "-to-datetime-parsing.html. You can safely remove this argument."
    ),
)


print("A")

wd = Path("~/Desktop/clay-v1-data/items/").expanduser()

ls_item = Item.from_file(
    wd / "landsat-c2l2-sr-LC09_L2SR_086107_20240311_20240312_02_T2_SR.json"
)
ls_idx = LandsatIndexer(ls_item)  # , assets=["coastal", "red", "green", "blue"])
ls_index = ls_idx.create_index()

s2_item = Item.from_file(wd / "sentinel-2-l2a-S2B_20HMF_20240309_0_L2A.json")
s2_idx = Sentinel2Indexer(s2_item)  # , assets=["nir", "red", "green", "blue"])
s2_index = s2_idx.create_index()

naip_item = Item.from_file(wd / "naip_m_4207009_ne_19_060_20211024.json")
naip_idx = NoStatsChipIndexer(naip_item)  # , assets=["asset"])
naip_index = naip_idx.create_index()

naip_item = Item.from_file(wd / "nz-auckland-2010-75mm-rgb-2193-BA32_1000_3212.json")
naip_idx = NoStatsChipIndexer(naip_item)  # , assets=["asset"])
naip_index = naip_idx.create_index()

print("B")
