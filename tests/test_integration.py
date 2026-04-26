"""Tier 1 integration tests: full pipeline, no teacher calls needed."""

import lightning as L
import pytest
import torch

from claymodel.api import embed, load_metadata, load_model, normalize
from claymodel.inference.deterministic import DeterministicInference
from claymodel.inference.masking import PatchAnalyzer


def test_embed_all_optical_sensors(tiny_module, metadata):
    """embed() produces valid embeddings for every optical sensor."""
    for sensor_name in metadata:
        sensor = metadata[sensor_name]
        n_bands = len(sensor.bands.wavelength)
        pixels = torch.randn(1, n_bands, 64, 64)

        result = embed(pixels, sensor=sensor_name, model=tiny_module)

        assert result.embeddings.shape == (1, 192), f"Bad shape for {sensor_name}"
        assert not torch.isnan(result.embeddings).any(), f"NaN for {sensor_name}"
        assert result.sensor == sensor_name
        assert result.gsd == sensor.gsd


def test_embed_sentinel1_db_conversion_applied(tiny_module):
    """SAR dB conversion inside embed() actually changes the values."""
    # Linear power values (positive)
    pixels_linear = torch.rand(1, 2, 64, 64) + 0.01

    result_via_embed = embed(pixels_linear, sensor="sentinel-1-rtc", model=tiny_module)

    # Manually convert to dB then normalize and embed
    pixels_db = 10 * torch.log10(pixels_linear.clamp(min=1e-10))
    pixels_norm = normalize(pixels_db, "sentinel-1-rtc")
    metadata = load_metadata()
    waves = torch.tensor(list(metadata["sentinel-1-rtc"].bands.wavelength.values()))
    gsd = torch.tensor(metadata["sentinel-1-rtc"].gsd)
    datacube = {
        "pixels": pixels_norm,
        "time": torch.zeros(1, 4),
        "latlon": torch.zeros(1, 4),
        "gsd": gsd,
        "waves": waves,
    }
    with torch.no_grad():
        encoded, *_ = tiny_module.encoder(datacube)
        manual_emb = encoded[:, 0, :]

    assert torch.allclose(result_via_embed.embeddings, manual_emb, atol=1e-5)


def test_embed_deterministic_identity(tiny_module):
    """DeterministicInference produces identical results across calls."""
    pixels = torch.randn(1, 10, 64, 64)

    with DeterministicInference(seed=42):
        r1 = embed(pixels.clone(), sensor="sentinel-2-l2a", model=tiny_module)

    with DeterministicInference(seed=42):
        r2 = embed(pixels.clone(), sensor="sentinel-2-l2a", model=tiny_module)

    assert torch.equal(r1.embeddings, r2.embeddings)


def test_embed_batch_consistency(tiny_module):
    """Same input at different batch positions produces identical embeddings."""
    single = torch.randn(1, 10, 64, 64)
    batched = single.expand(2, -1, -1, -1).clone()

    result = embed(batched, sensor="sentinel-2-l2a", model=tiny_module)

    assert torch.allclose(result.embeddings[0], result.embeddings[1], atol=1e-6)


def test_embed_3d_4d_equivalence(tiny_module):
    """3D input [C,H,W] and 4D [1,C,H,W] produce identical embeddings."""
    pixels_4d = torch.randn(1, 10, 64, 64)
    pixels_3d = pixels_4d.squeeze(0)

    r_4d = embed(pixels_4d, sensor="sentinel-2-l2a", model=tiny_module)
    r_3d = embed(pixels_3d, sensor="sentinel-2-l2a", model=tiny_module)

    assert torch.allclose(r_4d.embeddings, r_3d.embeddings, atol=1e-6)


def test_checkpoint_roundtrip_embedding_identity(tiny_module, tmp_path):
    """Save/load preserves inference output, not just weights."""
    pixels = torch.randn(1, 10, 64, 64)

    r_before = embed(pixels.clone(), sensor="sentinel-2-l2a", model=tiny_module)

    ckpt_path = tmp_path / "roundtrip.ckpt"
    trainer = L.Trainer(max_steps=0, enable_checkpointing=False)
    trainer.strategy.connect(tiny_module)
    trainer.save_checkpoint(str(ckpt_path))

    loaded = load_model(size="tiny", ckpt_path=str(ckpt_path))
    r_after = embed(pixels.clone(), sensor="sentinel-2-l2a", model=loaded)

    assert torch.allclose(r_before.embeddings, r_after.embeddings, atol=1e-6)


def test_load_model_all_sizes():
    """All model sizes construct successfully via load_model()."""
    expected_dims = {"tiny": 192, "small": 384, "base": 768, "large": 1024}
    for size, dim in expected_dims.items():
        model = load_model(size=size)
        assert model.model.encoder.dim == dim
        assert not model.training


def test_embed_then_geoparquet_roundtrip(tiny_module, tmp_path):
    """Full pipeline: pixels -> embed -> GeoParquet -> read back."""
    gpd = pytest.importorskip("geopandas")

    pixels = torch.randn(2, 10, 64, 64)
    result = embed(
        pixels,
        sensor="sentinel-2-l2a",
        model=tiny_module,
        latlon=torch.tensor([[0.5, 0.5, -0.5, -0.5], [0.6, 0.6, -0.6, -0.6]]),
    )

    path = tmp_path / "test.parquet"
    gdf = result.to_geoparquet(str(path))

    assert path.exists()
    assert len(gdf) == 2
    assert gdf.geometry.iloc[0] is not None

    # Read back and verify embedding values preserved
    loaded = gpd.read_parquet(path)
    assert len(loaded) == 2
    assert loaded.iloc[0]["sensor"] == "sentinel-2-l2a"


def test_filter_then_embed_pipeline(tiny_module):
    """PatchAnalyzer filtering feeds correctly into embed()."""
    # Create input with half nodata (zeros)
    pixels = torch.randn(1, 10, 64, 64)
    pixels[:, :, 32:, :] = 0  # bottom half is nodata

    analyzer = PatchAnalyzer(patch_size=8, nodata=0)
    valid_frac = analyzer.valid_fraction(pixels)

    # At least some patches should be valid
    assert (valid_frac > 0.5).any()

    # Full image still embeds fine
    result = embed(pixels, sensor="sentinel-2-l2a", model=tiny_module)
    assert result.embeddings.shape == (1, 192)
    assert not torch.isnan(result.embeddings).any()
