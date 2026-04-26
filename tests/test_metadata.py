"""Test bundled metadata loading (A3 verification)."""

from claymodel.api import load_metadata
from claymodel.metadata import Metadata, PlatformMetadata
from tests.conftest import make_metadata


def test_load_metadata_returns_metadata_type():
    metadata = make_metadata()
    assert isinstance(metadata, Metadata)


def test_metadata_has_sentinel2():
    metadata = make_metadata()
    assert "sentinel-2-l2a" in metadata


def test_metadata_sentinel2_bands():
    metadata = make_metadata()
    s2 = metadata["sentinel-2-l2a"]
    assert isinstance(s2, PlatformMetadata)
    assert s2.gsd == 10
    assert "blue" in s2.bands.wavelength
    assert s2.bands.wavelength["blue"] == 0.493


def test_metadata_has_sar():
    metadata = make_metadata()
    assert "sentinel-1-rtc" in metadata
    sar = metadata["sentinel-1-rtc"]
    assert sar.bands.wavelength["vv"] == 3.5
    assert sar.bands.wavelength["vh"] == 4.0
    assert sar.rgb_indices is None


def test_metadata_all_platforms_have_required_keys():
    metadata = make_metadata()
    for platform_name in metadata:
        platform = metadata[platform_name]
        assert isinstance(platform, PlatformMetadata)
        assert platform.gsd > 0, f"{platform_name} has invalid gsd"
        assert len(platform.bands.wavelength) > 0
        assert len(platform.bands.mean) > 0
        assert len(platform.bands.std) > 0


def test_load_metadata_api():
    metadata = load_metadata()
    assert "sentinel-2-l2a" in metadata


def test_load_metadata_custom_path(tmp_path):
    """Loading a custom metadata YAML should validate and return Metadata."""
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
my-sensor:
  band_order: [red, green, blue]
  gsd: 5.0
  bands:
    mean:
      red: 100.0
      green: 200.0
      blue: 150.0
    std:
      red: 50.0
      green: 60.0
      blue: 55.0
    wavelength:
      red: 0.665
      green: 0.56
      blue: 0.49
"""
    )
    metadata = load_metadata(str(custom))
    assert "my-sensor" in metadata
    assert metadata["my-sensor"].gsd == 5.0
    assert metadata["my-sensor"].bands.wavelength["red"] == 0.665
