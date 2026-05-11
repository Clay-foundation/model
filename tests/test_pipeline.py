"""Test inference pipeline utilities.

Tests the pure functions in claymodel.inference.pipeline that don't require
network access or optional dependencies (rustac, lazycogs).
"""

import math
from datetime import datetime

import numpy as np
import pytest
import torch

from claymodel.inference.pipeline import (
    _compute_chip_bbox,
    chip_array,
    encode_latlon,
    encode_timestamp,
)

rng = np.random.default_rng(42)


class TestEncodeTimestamp:
    def test_output_shape(self):
        dt = datetime(2024, 6, 15, 12, 0, 0)
        result = encode_timestamp(dt)
        assert result.shape == (4,)

    def test_dtype(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)
        result = encode_timestamp(dt)
        assert result.dtype == torch.float32

    def test_values_are_bounded(self):
        dt = datetime(2024, 7, 20, 14, 30, 0)
        result = encode_timestamp(dt)
        assert (result >= -1.0).all()
        assert (result <= 1.0).all()

    def test_midnight_hour_encoding(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)
        result = encode_timestamp(dt)
        # hour=0: sin(0)=0, cos(0)=1
        assert abs(result[2].item()) < 1e-6  # hour_sin ~ 0
        assert abs(result[3].item() - 1.0) < 1e-6  # hour_cos ~ 1

    def test_week_encoding_matches_manual(self):
        dt = datetime(2024, 6, 15, 10, 0, 0)
        result = encode_timestamp(dt)
        week = dt.isocalendar()[1] * 2 * math.pi / 52
        assert abs(result[0].item() - math.sin(week)) < 1e-6
        assert abs(result[1].item() - math.cos(week)) < 1e-6


class TestEncodeLatlon:
    def test_output_shape(self):
        result = encode_latlon(45.0, -93.0)
        assert result.shape == (4,)

    def test_dtype(self):
        result = encode_latlon(0.0, 0.0)
        assert result.dtype == torch.float32

    def test_equator_prime_meridian(self):
        result = encode_latlon(0.0, 0.0)
        # lat=0: sin(0)=0, cos(0)=1
        assert abs(result[0].item()) < 1e-6
        assert abs(result[1].item() - 1.0) < 1e-6
        # lon=0: sin(0)=0, cos(0)=1
        assert abs(result[2].item()) < 1e-6
        assert abs(result[3].item() - 1.0) < 1e-6

    def test_values_are_bounded(self):
        result = encode_latlon(89.5, -179.5)
        assert (result >= -1.0).all()
        assert (result <= 1.0).all()

    def test_matches_manual_computation(self):
        lat, lon = 37.3, -8.57
        result = encode_latlon(lat, lon)
        lat_rad = lat * math.pi / 180
        lon_rad = lon * math.pi / 180
        assert abs(result[0].item() - math.sin(lat_rad)) < 1e-6
        assert abs(result[1].item() - math.cos(lat_rad)) < 1e-6
        assert abs(result[2].item() - math.sin(lon_rad)) < 1e-6
        assert abs(result[3].item() - math.cos(lon_rad)) < 1e-6


class TestChipArray:
    def test_basic_chipping(self):
        pixels = rng.random((10, 512, 512)).astype(np.float32)
        chips = chip_array(pixels, chip_size=256)
        assert len(chips) == 4  # 2x2 grid
        for chip, _row, _col in chips:
            assert chip.shape == (10, 256, 256)

    def test_chip_indices(self):
        pixels = rng.random((10, 512, 768)).astype(np.float32)
        chips = chip_array(pixels, chip_size=256)
        # 2 rows x 3 cols = 6 chips
        assert len(chips) == 6
        indices = [(r, c) for _, r, c in chips]
        assert (0, 0) in indices
        assert (0, 1) in indices
        assert (0, 2) in indices
        assert (1, 0) in indices
        assert (1, 1) in indices
        assert (1, 2) in indices

    def test_drops_edge_pixels(self):
        # 300x300 with chip_size=256 -> only 1 chip (edges dropped)
        pixels = rng.random((10, 300, 300)).astype(np.float32)
        chips = chip_array(pixels, chip_size=256)
        assert len(chips) == 1

    def test_exact_fit(self):
        pixels = rng.random((3, 256, 256)).astype(np.float32)
        chips = chip_array(pixels, chip_size=256)
        assert len(chips) == 1
        np.testing.assert_array_equal(chips[0][0], pixels)

    def test_chip_size_not_divisible_by_patch_size(self):
        pixels = rng.random((10, 512, 512)).astype(np.float32)
        with pytest.raises(ValueError, match="divisible by patch_size"):
            chip_array(pixels, chip_size=100, patch_size=8)

    def test_image_too_small(self):
        pixels = rng.random((10, 100, 100)).astype(np.float32)
        with pytest.raises(ValueError, match="smaller than chip_size"):
            chip_array(pixels, chip_size=256)

    def test_custom_chip_size(self):
        pixels = rng.random((5, 128, 128)).astype(np.float32)
        chips = chip_array(pixels, chip_size=64)
        assert len(chips) == 4  # 2x2

    def test_chips_dont_overlap(self):
        pixels = np.arange(10 * 512 * 512, dtype=np.float32).reshape(10, 512, 512)
        chips = chip_array(pixels, chip_size=256)
        # All chips should have unique data
        for i in range(len(chips)):
            for j in range(i + 1, len(chips)):
                assert not np.array_equal(chips[i][0], chips[j][0])


class TestComputeChipBbox:
    def test_first_chip(self):
        scene_bbox = [500000.0, 4900000.0, 510000.0, 4910000.0]
        bbox = _compute_chip_bbox(
            scene_bbox, row=0, col=0, chip_size=256, resolution=10.0, full_height=1000
        )
        # minx = 500000, maxy = 4910000
        # chip_extent = 256 * 10 = 2560
        assert bbox[0] == pytest.approx(500000.0)
        assert bbox[3] == pytest.approx(4910000.0)
        assert bbox[2] - bbox[0] == pytest.approx(2560.0)
        assert bbox[3] - bbox[1] == pytest.approx(2560.0)

    def test_second_row_first_col(self):
        scene_bbox = [500000.0, 4900000.0, 510000.0, 4910000.0]
        bbox = _compute_chip_bbox(
            scene_bbox, row=1, col=0, chip_size=256, resolution=10.0, full_height=1000
        )
        chip_extent = 256 * 10.0
        assert bbox[3] == pytest.approx(4910000.0 - chip_extent)

    def test_first_row_second_col(self):
        scene_bbox = [500000.0, 4900000.0, 510000.0, 4910000.0]
        bbox = _compute_chip_bbox(
            scene_bbox, row=0, col=1, chip_size=256, resolution=10.0, full_height=1000
        )
        chip_extent = 256 * 10.0
        assert bbox[0] == pytest.approx(500000.0 + chip_extent)
