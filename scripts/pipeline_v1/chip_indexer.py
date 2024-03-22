import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pyarrow as pa
import stackstac
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


class ChipIndexer:
    shape_source: str = "proj:shape"

    def __init__(
        self,
        item: Item,
        chip_size: int = 256,
        needs_stats=True,
    ) -> None:
        self.item = item
        self.chip_size = chip_size
        self.needs_stats = needs_stats

    def get_item_shape(self) -> list:
        """
        Get shape of hightest resolution band.
        """
        shape = None
        if self.shape_source in self.item.properties:
            shape = self.item.properties[self.shape_source]
        else:
            for asset in self.item.assets.values():
                if self.shape_source not in asset.extra_fields:
                    continue
                if shape is None or shape[0] < asset.extra_fields[self.shape_source][0]:
                    shape = asset.extra_fields[self.shape_source]

        if shape is None:
            raise ValueError(f"No shape found in item {self.item}")

        print("Shape", shape)
        return shape

    def get_stats(self, chip_index_x: int, chip_index_y: int) -> Tuple[float, float]:
        if self.needs_stats:
            raise NotImplementedError()
        return 0.0, 0.0

    def create_index(self) -> None:
        index = []
        chipid = 0

        shape = self.get_item_shape()

        for y in range(0, shape[1], self.chip_size):
            for x in range(0, shape[0], self.chip_size):
                cloud_cover_percentage, nodata_percentage = self.get_stats(x, y)
                row = [
                    Path(self.item.get_self_href()).name,
                    f"{self.item.id}-{chipid}",
                    x,
                    y,
                    cloud_cover_percentage,
                    nodata_percentage,
                ]
                index.append(row)
                chipid += 1

        return pa.table(
            [
                pa.array([dat[0] for dat in index], type=pa.string()),
                pa.array([dat[1] for dat in index], type=pa.string()),
                pa.array([dat[2] for dat in index], type=pa.int16()),
                pa.array([dat[3] for dat in index], type=pa.int16()),
                pa.array([dat[4] for dat in index], type=pa.float32()),
                pa.array([dat[5] for dat in index], type=pa.float32()),
            ],
            names=[
                "stac_item",
                "chipid",
                "chip_index_x",
                "chip_index_y",
                "cloud_cover_percentage",
                "nodata_percentage",
            ],
        )


class LandsatIndexer(ChipIndexer):
    _qa = None

    @property
    def qa(self):
        if self._qa is None:
            print("Loading qa band")
            self.item.assets["qa_pixel"].href = self.item.assets[
                "qa_pixel"
            ].extra_fields["alternate"]["s3"]["href"]
            stack = stackstac.stack(
                [self.item], assets=["qa_pixel"], dtype="uint16", fill_value=0
            )
            self._qa = stack.compute()[0, 0, :, :].to_numpy()
        return self._qa

    def get_stats(self, x: int, y: int) -> Tuple[float, float]:
        qa = self.qa[y : (y + self.chip_size), x : (x + self.chip_size)]

        # Bit 1 is dilated cloud, 3 is cloud, 4 is cloud shadow.
        nodata_byte = np.array(1 << 0, dtype=qa.dtype)
        dilated_cloud_byte = np.array(1 << 1, dtype=qa.dtype)
        cloud_byte = np.array(1 << 3, dtype=qa.dtype)
        shadow_byte = np.array(1 << 4, dtype=qa.dtype)

        nodata_mask = np.bitwise_and(qa, nodata_byte)
        dilated_cloud = np.bitwise_and(qa, dilated_cloud_byte)
        cloud = np.bitwise_and(qa, cloud_byte)
        shadow = np.bitwise_and(qa, shadow_byte)

        layer_clouds = (dilated_cloud | cloud | shadow).astype(dtype="bool")

        cloud_percentage = np.sum(layer_clouds) / qa.size
        nodata_percentage = np.sum(nodata_mask) / qa.size

        return cloud_percentage, nodata_percentage


class Sentinel2Indexer(ChipIndexer):
    scl_filter = [1, 3, 8, 9, 10]
    nodata_value = 0

    _scl = None

    @property
    def scl(self):
        if self._scl is None:
            print("Loading scl band")
            stack = stackstac.stack(
                [self.item], assets=["scl"], dtype="uint8", fill_value=0, resolution=10
            )
            self._scl = stack.compute()[0, 0, :, :].to_numpy()
        return self._scl

    def get_stats(self, x: int, y: int) -> Tuple[float, float]:
        scl = self.scl[y : (y + self.chip_size), x : (x + self.chip_size)]

        cloud_percentage = int(np.isin(scl, self.scl_filter).sum()) / scl.size

        nodata_percentage = np.sum(scl == self.nodata_value) / scl.size

        return cloud_percentage, nodata_percentage
