"""Test the high-level API."""

import numpy as np
import pytest
import torch

from claymodel.api import EmbeddingResult, embed, load_metadata, load_model, normalize
from claymodel.model import Encoder
from tests.conftest import make_tiny_encoder


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
    """Test embed() end-to-end with a tiny encoder (no checkpoint needed)."""
    encoder = make_tiny_encoder()
    encoder.eval()
    pixels = torch.randn(1, 10, 64, 64)
    result = embed(pixels, sensor="sentinel-2-l2a", model=encoder)

    assert result.embeddings.shape == (1, 192)  # tiny dim=192
    assert result.sensor == "sentinel-2-l2a"
    assert result.gsd == 10.0


def test_embed_numerical_identity():
    """Verify embed() produces identical output to manual datacube path."""
    metadata = load_metadata()
    sensor = "sentinel-2-l2a"
    encoder = make_tiny_encoder()
    encoder.eval()

    raw_pixels = torch.randn(1, 10, 64, 64)

    # Path 1: via embed()
    result = embed(raw_pixels.clone(), sensor=sensor, model=encoder)

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
        encoded, *_ = encoder(datacube)
        manual_emb = encoded[:, 0, :]

    assert torch.allclose(result.embeddings, manual_emb, atol=1e-6)


def test_embed_requires_model_or_ckpt():
    """embed() should raise ValueError if neither model nor ckpt provided."""
    pixels = torch.randn(1, 10, 64, 64)
    with pytest.raises(ValueError, match="model or ckpt_path"):
        embed(pixels, sensor="sentinel-2-l2a")


def test_embed_with_numpy_array():
    """embed() should accept numpy arrays."""
    encoder = make_tiny_encoder()
    encoder.eval()
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal((1, 10, 64, 64)).astype(np.float32)
    result = embed(pixels, sensor="sentinel-2-l2a", model=encoder)
    assert result.embeddings.shape == (1, 192)


def test_embed_auto_unsqueeze_3d():
    """3D tensor [C, H, W] should be auto-unsqueezed to [1, C, H, W]."""
    encoder = make_tiny_encoder()
    encoder.eval()
    pixels = torch.randn(10, 64, 64)  # 3D — no batch dim
    result = embed(pixels, sensor="sentinel-2-l2a", model=encoder)
    assert result.embeddings.shape == (1, 192)


def test_embed_3d_numpy():
    """3D numpy array should be auto-unsqueezed."""
    encoder = make_tiny_encoder()
    encoder.eval()
    rng = np.random.default_rng(42)
    pixels = rng.standard_normal((10, 64, 64)).astype(np.float32)
    result = embed(pixels, sensor="sentinel-2-l2a", model=encoder)
    assert result.embeddings.shape == (1, 192)


def test_embed_invalid_input_type():
    """embed() with invalid type should raise TypeError."""
    with pytest.raises(TypeError, match="Tensor or ndarray"):
        embed(42, sensor="sentinel-2-l2a")  # ty: ignore[invalid-argument-type]


def test_embed_unknown_sensor():
    """embed() with unknown sensor should raise ValueError."""
    encoder = make_tiny_encoder()
    encoder.eval()
    with pytest.raises(ValueError, match="Unknown sensor"):
        embed(torch.randn(1, 3, 64, 64), sensor="fake-sensor", model=encoder)


def test_embed_sentinel1_db_conversion():
    """For sentinel-1-rtc, embed() should convert linear power to dB."""
    encoder = make_tiny_encoder()
    encoder.eval()
    # Use linear power values (positive, raw)
    pixels = torch.rand(1, 2, 64, 64) + 0.01  # positive values
    result = embed(pixels, sensor="sentinel-1-rtc", model=encoder)
    assert result.embeddings.shape == (1, 192)
    assert not torch.isnan(result.embeddings).any()


def test_normalize_output_device_matches_input():
    """Normalized output should be on the same device as input."""
    pixels = torch.randn(1, 10, 64, 64)
    result = normalize(pixels, "sentinel-2-l2a")
    assert result.device == pixels.device


def test_load_model_no_checkpoint():
    """load_model without checkpoint returns a working encoder with random weights."""
    encoder = load_model(size="tiny")
    assert isinstance(encoder, Encoder)
    assert not encoder.training  # eval mode
    assert encoder.mask_ratio == 0.0
    assert encoder.shuffle is False


def test_load_model_with_checkpoint(tmp_path):
    """Save and reload a tiny model checkpoint."""
    import lightning as L

    from claymodel.module import ClayMAEModule
    from tests.conftest import _bundled_metadata_path

    # Create a full module to save a checkpoint
    original = ClayMAEModule(
        model_size="tiny",
        mask_ratio=0.0,
        shuffle=False,
        metadata_path=_bundled_metadata_path(),
        teacher="samvit_base_patch16.sa1b",
    )
    ckpt_path = tmp_path / "tiny.ckpt"

    trainer = L.Trainer(max_steps=0, enable_checkpointing=False)
    trainer.strategy.connect(original)
    trainer.save_checkpoint(str(ckpt_path))

    # Reload as Encoder (no teacher download)
    loaded = load_model(size="tiny", ckpt_path=str(ckpt_path))
    assert isinstance(loaded, Encoder)
    assert not loaded.training

    # Encoder weights should match
    original_encoder = original.model.encoder
    for key in loaded.state_dict():
        if key in original_encoder.state_dict():
            assert torch.allclose(
                original_encoder.state_dict()[key],
                loaded.state_dict()[key],
            ), f"Mismatch in {key}"
