"""Test PatchAnalyzer (C2 verification)."""

import torch

from claymodel.inference.masking import PatchAnalyzer


def test_valid_fraction_all_valid():
    analyzer = PatchAnalyzer(patch_size=8, nodata=0)
    pixels = torch.ones(1, 3, 64, 64)  # All valid (non-zero)
    vf = analyzer.valid_fraction(pixels)
    assert vf.shape == (1, 64)  # 64 patches = (64/8)^2
    assert torch.allclose(vf, torch.ones_like(vf))


def test_valid_fraction_all_nodata():
    analyzer = PatchAnalyzer(patch_size=8, nodata=0)
    pixels = torch.zeros(1, 3, 64, 64)  # All nodata
    vf = analyzer.valid_fraction(pixels)
    assert torch.allclose(vf, torch.zeros_like(vf))


def test_valid_fraction_half_nodata():
    analyzer = PatchAnalyzer(patch_size=8, nodata=0)
    pixels = torch.ones(1, 3, 64, 64)
    pixels[:, :, :32, :] = 0  # Top half is nodata
    vf = analyzer.valid_fraction(pixels)
    # Top patches should be 0, bottom patches should be 1
    assert vf.min() == 0.0
    assert vf.max() == 1.0


def test_valid_fraction_nan_nodata():
    analyzer = PatchAnalyzer(patch_size=8, nodata="nan")
    pixels = torch.ones(1, 3, 64, 64)
    pixels[:, 0, :8, :8] = float("nan")  # One patch has NaN in one band
    vf = analyzer.valid_fraction(pixels)
    # The NaN patch should have fraction < 1.0
    assert vf.min() < 1.0


def test_cloud_fraction():
    analyzer = PatchAnalyzer(patch_size=8)
    # SCL band: 8 = cloud medium prob, 9 = cloud high prob, 10 = cirrus
    scl = torch.zeros(1, 64, 64, dtype=torch.long)
    scl[:, :8, :8] = 9  # One patch is fully cloudy
    cf = analyzer.cloud_fraction(scl)
    assert cf.shape == (1, 64)
    assert cf.max() == 1.0  # Fully cloudy patch
    assert cf.min() == 0.0  # Clear patches


def test_filter_chips_passes():
    analyzer = PatchAnalyzer(patch_size=8, nodata=0)
    pixels = torch.ones(2, 3, 64, 64)  # All valid
    mask = analyzer.filter_chips(pixels, min_valid=0.5)
    assert mask.shape == (2,)
    assert mask.all()  # Both chips pass


def test_filter_chips_rejects():
    analyzer = PatchAnalyzer(patch_size=8, nodata=0)
    pixels = torch.zeros(2, 3, 64, 64)  # All nodata
    mask = analyzer.filter_chips(pixels, min_valid=0.5)
    assert not mask.any()  # Both chips rejected


def test_filter_chips_with_cloud():
    analyzer = PatchAnalyzer(patch_size=8, nodata=0)
    pixels = torch.ones(1, 3, 64, 64)
    scl = torch.full((1, 64, 64), 9, dtype=torch.long)  # Fully cloudy
    mask = analyzer.filter_chips(pixels, min_valid=0.5, scl_band=scl, max_cloud=0.5)
    assert not mask.any()  # Rejected due to clouds
