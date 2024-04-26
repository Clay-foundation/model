import math
from typing import Iterator, List

import numpy as np
from torch.utils.data.sampler import Sampler


class ClaySampler(Sampler):
    def __init__(
        self,
        data: List[str],
        platforms: List[str],
        batch_size: int,
        cube_size: int = 128,
    ) -> None:
        self.platforms = platforms
        self.batch_size = batch_size
        self.cube_size = cube_size
        self.chip_count = len(data) * cube_size

        # Split data into platforms
        self.cubes_per_platform = {}
        self.batch_count_per_platform = {}
        for platform in self.platforms:
            self.cubes_per_platform[platform] = [
                cube for cube in data if platform in cube
            ]
            self.batch_count_per_platform[platform] = math.floor(
                len(self.cubes_per_platform[platform])
                * self.cube_size
                / self.batch_size
            )
        self._length = max(self.batch_count_per_platform.values()) * len(self.platforms)

    def __len__(self) -> int:
        return self._length

    def get_cube(self, platform: str, cube_idx: int):
        return

    def get_batch(self, batchnr: int):
        # Get platform for this batch
        platform = self.platforms[batchnr % len(self.platforms)]
        # Get cubes for this plaform
        cubes = self.cubes_per_platform[platform]
        # Get batch number for this platform
        iteration_count = math.floor(batchnr / len(self.platforms))
        platform_batchnr = iteration_count % self.batch_count_per_platform[platform]
        # Get cubes for this batch
        cubes_idx_low = math.floor(
            (platform_batchnr * self.batch_size) / self.cube_size
        )
        cubes_range_index_low = (platform_batchnr * self.batch_size) % self.cube_size
        cubes_idx_high = math.floor(
            ((platform_batchnr + 1) * self.batch_size) / self.cube_size
        )
        cubes_range_index_high = cubes_range_index_low + self.batch_size
        # Load cubes data
        cubes = []
        for cube_idx in range(cubes_idx_low, cubes_idx_high + 1):
            cube_path = self.cubes_per_platform[platform][cube_idx]
            cubes.append(np.load(cube_path))
        # Combine cubes to batch
        CUBE_KEYS = [
            "pixels",
            "lon_norm",
            "lat_norm",
            "week_norm",
            "hour_norm",
        ]
        result = []
        for key in CUBE_KEYS:
            result.append(
                np.vstack([cube[key] for cube in cubes])[
                    cubes_range_index_low:cubes_range_index_high
                ]
            )
        return result

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def __iter__(self) -> Iterator[int]:
        for batchnr in range(self._length):
            yield self.get_batch(batchnr)
