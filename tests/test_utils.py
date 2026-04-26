"""Test utility functions."""

import tempfile

import torch

from claymodel.utils import (
    load_encoder_weights,
    posemb_sincos_1d,
    posemb_sincos_2d,
    posemb_sincos_2d_with_gsd,
)
from tests.conftest import make_tiny_encoder


def test_posemb_sincos_2d_shape():
    pe = posemb_sincos_2d(h=8, w=8, dim=64)
    assert pe.shape == (64, 64)  # 8*8 positions, 64 dims


def test_posemb_sincos_2d_with_gsd_shape():
    gsd = torch.tensor(10.0)
    pe = posemb_sincos_2d_with_gsd(h=8, w=8, dim=64, gsd=gsd)
    assert pe.shape == (64, 64)


def test_posemb_sincos_1d_shape():
    waves = torch.tensor([0.49, 0.56, 0.66])
    pe = posemb_sincos_1d(waves, dim=128)
    assert pe.shape == (3, 128)


def test_posemb_sincos_1d_integer_input():
    pe = posemb_sincos_1d(5, dim=64)
    assert pe.shape == (5, 64)


def test_load_encoder_weights():
    """Test the shared weight loading utility with a fake checkpoint."""
    # Create source encoder and save as checkpoint
    source = make_tiny_encoder()
    state_dict = {"model.encoder." + k: v for k, v in source.state_dict().items()}
    ckpt = {"state_dict": state_dict}

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        torch.save(ckpt, f.name)

        # Create target encoder and load weights
        target = make_tiny_encoder()
        loaded, skipped = load_encoder_weights(target, f.name, freeze=True)

    # Verify weights match
    for name, param in source.state_dict().items():
        assert torch.equal(param, target.state_dict()[name]), f"Mismatch in {name}"

    assert len(loaded) > 0
    assert len(skipped) == 0

    # Verify frozen
    for param in target.parameters():
        assert not param.requires_grad


def test_load_encoder_weights_no_freeze():
    """With freeze=False, parameters should remain trainable."""
    source = make_tiny_encoder()
    state_dict = {"model.encoder." + k: v for k, v in source.state_dict().items()}
    ckpt = {"state_dict": state_dict}

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        torch.save(ckpt, f.name)
        target = make_tiny_encoder()
        load_encoder_weights(target, f.name, freeze=False)

    # Parameters should still require grad
    for param in target.parameters():
        assert param.requires_grad


def test_load_encoder_weights_skips_mismatched():
    """Keys with size mismatch should be skipped, not error."""
    source = make_tiny_encoder()
    state_dict = {"model.encoder." + k: v for k, v in source.state_dict().items()}

    # Add a key with wrong size
    state_dict["model.encoder.fake_param"] = torch.randn(999)
    ckpt = {"state_dict": state_dict}

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        torch.save(ckpt, f.name)
        target = make_tiny_encoder()
        loaded, skipped = load_encoder_weights(target, f.name)

    assert "fake_param" in skipped
    assert len(loaded) > 0


def test_posemb_sincos_2d_bad_dim():
    """Dimension not divisible by 4 should raise AssertionError."""
    try:
        posemb_sincos_2d(h=8, w=8, dim=63)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
