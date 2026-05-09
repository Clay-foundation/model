"""Test the high-level API."""

import warnings
from importlib.resources import files
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from claymodel.api import EmbeddingResult, embed, load_metadata, load_model, normalize
from claymodel.module import ClayMAEModule


def _bundled_metadata_path():
    return str(files("claymodel").joinpath("configs/metadata.yaml"))


def _make_tiny_module():
    """Create a tiny ClayMAEModule for testing. Downloads teacher on first call."""
    model = ClayMAEModule(
        model_size="tiny",
        mask_ratio=0.0,
        shuffle=False,
        metadata_path=_bundled_metadata_path(),
        teacher="samvit_base_patch16.sa1b",
    )
    model.eval()
    model.model.encoder.mask_ratio = 0.0
    model.model.encoder.shuffle = False
    return model


def test_normalize_sentinel2():
    metadata = load_metadata()
    pixels = torch.ones(1, 10, 64, 64) * 1000.0
    result = normalize(pixels, "sentinel-2-l2a", metadata=metadata)

    # After z-score normalization, result should not be all 1000
    assert result.shape == pixels.shape
    assert not torch.allclose(result, pixels)


def test_normalize_preserves_shape():
    pixels = torch.randn(2, 10, 64, 64)
    result = normalize(pixels, "sentinel-2-l2a")
    assert result.shape == (2, 10, 64, 64)


def test_normalize_unknown_sensor():
    pixels = torch.randn(1, 3, 64, 64)
    with pytest.raises(ValueError, match="nonexistent-sensor"):
        normalize(pixels, "nonexistent-sensor")


def test_normalize_different_sensors():
    metadata = load_metadata()
    for sensor in ["sentinel-2-l2a", "naip", "linz", "sentinel-1-rtc"]:
        n_bands = len(metadata[sensor].bands.wavelength)
        pixels = torch.randn(1, n_bands, 64, 64)
        result = normalize(pixels, sensor, metadata=metadata)
        assert result.shape == pixels.shape


def test_embedding_result_shape():
    emb = EmbeddingResult(
        embeddings=torch.randn(4, 1024),
        sensor="sentinel-2-l2a",
        gsd=10.0,
    )
    assert emb.shape == (4, 1024)


def test_embedding_result_metadata():
    emb = EmbeddingResult(
        embeddings=torch.randn(1, 192),
        sensor="naip",
        gsd=1.0,
        metadata={"source": "test"},
    )
    assert emb.sensor == "naip"
    assert emb.gsd == 1.0
    assert emb.metadata["source"] == "test"


def test_embed_with_preloaded_model():
    """Test embed() end-to-end with a tiny model (no checkpoint needed)."""
    model = _make_tiny_module()
    pixels = torch.randn(1, 10, 64, 64)
    result = embed(pixels, sensor="sentinel-2-l2a", model=model)

    assert result.embeddings.shape == (1, 192)  # tiny dim=192
    assert result.sensor == "sentinel-2-l2a"
    assert result.gsd == 10.0


def test_embed_numerical_identity():
    """Verify embed() produces identical output to manual datacube path."""
    metadata = load_metadata()
    sensor = "sentinel-2-l2a"
    model = _make_tiny_module()

    raw_pixels = torch.randn(1, 10, 64, 64)

    # Path 1: via embed()
    result = embed(raw_pixels.clone(), sensor=sensor, model=model)

    # Path 2: manual datacube
    normalized = normalize(raw_pixels.clone(), sensor, metadata=metadata)
    datacube = {
        "pixels": normalized,
        "time": torch.zeros(1, 4),
        "latlon": torch.zeros(1, 4),
        "gsd": torch.tensor(float(metadata[sensor].gsd)),
        "waves": torch.tensor(list(metadata[sensor].bands.wavelength.values())),
    }
    with torch.no_grad():
        encoded, *_ = model.encoder(datacube)
        manual_emb = encoded[:, 0, :]

    assert torch.allclose(result.embeddings, manual_emb, atol=1e-6)


def test_embed_requires_model_or_ckpt():
    """embed() should raise ValueError if neither model nor ckpt provided."""
    pixels = torch.randn(1, 10, 64, 64)
    with pytest.raises(ValueError, match="model or ckpt_path"):
        embed(pixels, sensor="sentinel-2-l2a")


def test_embed_with_numpy_array():
    """embed() should accept numpy arrays."""
    model = _make_tiny_module()
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal((1, 10, 64, 64)).astype(np.float32)
    result = embed(pixels, sensor="sentinel-2-l2a", model=model)
    assert result.embeddings.shape == (1, 192)


def test_embed_auto_unsqueeze_3d():
    """3D tensor [C, H, W] should be auto-unsqueezed to [1, C, H, W]."""
    model = _make_tiny_module()
    pixels = torch.randn(10, 64, 64)  # 3D — no batch dim
    result = embed(pixels, sensor="sentinel-2-l2a", model=model)
    assert result.embeddings.shape == (1, 192)


def test_embed_3d_numpy():
    """3D numpy array should be auto-unsqueezed."""
    model = _make_tiny_module()
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal((10, 64, 64)).astype(np.float32)
    result = embed(pixels, sensor="sentinel-2-l2a", model=model)
    assert result.embeddings.shape == (1, 192)


def test_embed_invalid_input_type():
    """embed() with invalid type should raise TypeError."""
    with pytest.raises(TypeError, match="Tensor, ndarray, or file path"):
        embed(42, sensor="sentinel-2-l2a")  # ty: ignore[invalid-argument-type]


def test_embed_unknown_sensor():
    """embed() with unknown sensor should raise ValueError."""
    model = _make_tiny_module()
    with pytest.raises(ValueError, match="Unknown sensor"):
        embed(torch.randn(1, 3, 64, 64), sensor="fake-sensor", model=model)


def test_embed_sentinel1_db_conversion():
    """For sentinel-1-rtc, embed() should convert linear power to dB."""
    model = _make_tiny_module()
    # Use linear power values (positive, raw)
    pixels = torch.rand(1, 2, 64, 64) + 0.01  # positive values
    result = embed(pixels, sensor="sentinel-1-rtc", model=model)
    assert result.embeddings.shape == (1, 192)
    assert not torch.isnan(result.embeddings).any()


def test_embed_quality_warns_when_probe_missing():
    """quality=True should warn, not error, when probe weights missing."""
    model = _make_tiny_module()
    pixels = torch.randn(1, 10, 64, 64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = embed(pixels, sensor="sentinel-2-l2a", model=model, quality=True)
        assert result.embeddings.shape == (1, 192)
        # Should have warned about missing probe
        probe_warned = any("ELLE" in str(w_.message) or "probe" in str(w_.message) for w_ in w)
        assert probe_warned


def test_normalize_output_device_matches_input():
    """Normalized output should be on the same device as input."""
    pixels = torch.randn(1, 10, 64, 64)
    result = normalize(pixels, "sentinel-2-l2a")
    assert result.device == pixels.device


def test_load_model_no_checkpoint():
    """load_model without checkpoint returns a working model with random weights."""
    model = load_model(size="tiny")
    assert isinstance(model, ClayMAEModule)
    assert not model.training  # eval mode
    assert model.model.encoder.mask_ratio == 0.0
    assert model.model.encoder.shuffle is False


def test_load_model_with_checkpoint(tmp_path):
    """Save and reload a tiny model checkpoint."""
    import lightning as L

    original = _make_tiny_module()
    ckpt_path = tmp_path / "tiny.ckpt"

    # Use Lightning's Trainer to save a proper checkpoint
    trainer = L.Trainer(max_steps=0, enable_checkpointing=False)
    trainer.strategy.connect(original)
    trainer.save_checkpoint(str(ckpt_path))

    # Reload
    loaded = load_model(size="tiny", ckpt_path=str(ckpt_path))
    assert isinstance(loaded, ClayMAEModule)
    assert not loaded.training

    # Weights should match
    for key in original.state_dict():
        if "teacher" not in key:
            assert torch.allclose(
                original.state_dict()[key],
                loaded.state_dict()[key],
            ), f"Mismatch in {key}"


def test_to_geoparquet_with_geometry(tmp_path):
    """to_geoparquet should produce a valid file with point geometry."""
    pytest.importorskip("geopandas")

    latlon = torch.tensor([[37.0, -122.0], [38.0, -121.0], [39.0, -120.0]])
    emb = EmbeddingResult(
        embeddings=torch.randn(3, 64),
        sensor="sentinel-2-l2a",
        gsd=10.0,
        metadata={"latlon": latlon},
    )
    path = tmp_path / "test.parquet"
    gdf = cast(Any, emb.to_geoparquet(str(path)))  # noqa: TC006

    assert path.exists()
    assert len(gdf) == 3
    assert gdf.geometry.iloc[0] is not None


def test_to_geoparquet_without_geometry(tmp_path):
    """to_geoparquet should work with no latlon metadata."""
    pytest.importorskip("geopandas")

    emb = EmbeddingResult(
        embeddings=torch.randn(2, 64),
        sensor="naip",
        gsd=1.0,
    )
    path = tmp_path / "test_no_geo.parquet"
    gdf = cast(Any, emb.to_geoparquet(str(path)))  # noqa: TC006

    assert path.exists()
    assert len(gdf) == 2


def test_to_geoparquet_import_error():
    """Raise ImportError with helpful message if geopandas missing."""
    emb = EmbeddingResult(embeddings=torch.randn(1, 64))

    with patch.dict("sys.modules", {"geopandas": None}):
        with pytest.raises(ImportError, match="geopandas"):
            emb.to_geoparquet("out.parquet")
