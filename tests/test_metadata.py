"""Test bundled metadata loading (A3 verification)."""

from claymodel.api import load_metadata
from tests.conftest import make_metadata


def test_load_metadata_returns_box():
    metadata = make_metadata()
    assert hasattr(metadata, "keys")


def test_metadata_has_sentinel2():
    metadata = make_metadata()
    assert "sentinel-2-l2a" in metadata


def test_metadata_sentinel2_bands():
    metadata = make_metadata()
    s2 = metadata["sentinel-2-l2a"]
    assert s2.gsd == 10
    assert "blue" in s2.bands.wavelength
    assert s2.bands.wavelength.blue == 0.493


def test_metadata_has_sar():
    metadata = make_metadata()
    assert "sentinel-1-rtc" in metadata
    sar = metadata["sentinel-1-rtc"]
    assert sar.bands.wavelength.vv == 3.5
    assert sar.bands.wavelength.vh == 4.0


def test_metadata_all_platforms_have_required_keys():
    metadata = make_metadata()
    for platform_name in metadata:
        platform = metadata[platform_name]
        assert "gsd" in platform, f"{platform_name} missing gsd"
        assert "bands" in platform, f"{platform_name} missing bands"
        assert "wavelength" in platform.bands, f"{platform_name} missing wavelength"
        assert "mean" in platform.bands, f"{platform_name} missing mean"
        assert "std" in platform.bands, f"{platform_name} missing std"


def test_load_metadata_api():
    metadata = load_metadata()
    assert "sentinel-2-l2a" in metadata
