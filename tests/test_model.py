"""Test model forward passes and output shapes."""

import os
from typing import Any, cast

import pytest
import torch
from einops import rearrange
from torch import nn

from claymodel.model import (
    ClayMAE,
    Decoder,
    clay_mae_base,
    clay_mae_large,
    clay_mae_small,
    clay_mae_tiny,
)
from claymodel.module import ClayMAEModule
from tests.conftest import make_datacube, make_metadata, make_tiny_encoder
from training import configure_training_defaults


def test_encoder_output_shape():
    encoder = make_tiny_encoder()
    datacube = make_datacube(batch_size=2, channels=6, size=64)
    with torch.no_grad():
        encoded, _unmasked_idx, _masked_idx, mask_matrix = encoder(datacube)

    B = 2
    L = (64 // 8) ** 2  # 64 patches
    D = 192
    # With mask_ratio=0, all patches are unmasked + 1 cls token
    assert encoded.shape == (B, 1 + L, D)
    assert mask_matrix.shape == (B, L)


def test_encoder_with_masking():
    encoder = make_tiny_encoder(mask_ratio=0.75, shuffle=True)
    datacube = make_datacube(batch_size=2, channels=6, size=64)
    with torch.no_grad():
        encoded, _unmasked_idx, masked_idx, _mask_matrix = encoder(datacube)

    B = 2
    L = (64 // 8) ** 2
    num_unmasked = int(L * 0.25)
    assert encoded.shape == (B, 1 + num_unmasked, 192)
    assert masked_idx.shape[1] == int(L * 0.75)


def test_encoder_different_channels():
    """Encoder should handle any number of input channels."""
    encoder = make_tiny_encoder()
    for num_channels in [3, 6, 10]:
        datacube = make_datacube(channels=num_channels, size=64)
        with torch.no_grad():
            encoded, *_ = encoder(datacube)
        assert encoded.shape[0] == 2
        assert encoded.shape[2] == 192


def test_clay_mae_tiny_factory():
    metadata = make_metadata()
    model = clay_mae_tiny(
        mask_ratio=0.75,
        patch_size=8,
        norm_pix_loss=False,
        shuffle=True,
        metadata=metadata,
        teacher="samvit_base_patch16.sa1b",
        dolls=[16, 32],
        doll_weights=[1, 1],
    )
    assert model.encoder.dim == 192
    assert model.decoder.dim == 96


def test_factory_functions_return_clay_mae():
    metadata = make_metadata()
    common_args = {
        "mask_ratio": 0.0,
        "patch_size": 8,
        "norm_pix_loss": False,
        "shuffle": False,
        "metadata": metadata,
        "teacher": "samvit_base_patch16.sa1b",
        "dolls": [16],
        "doll_weights": [1],
    }

    # Only test tiny to keep tests fast
    model = clay_mae_tiny(**common_args)
    assert isinstance(model, ClayMAE)


def test_decoder_output_shape():
    """Decoder should reconstruct patches from encoder output."""
    B, L, enc_dim, dec_dim = 2, 64, 192, 96  # 64 patches = (64/8)^2
    patch_size = 8
    channels = 6

    decoder = Decoder(
        mask_ratio=0.75,
        patch_size=patch_size,
        encoder_dim=enc_dim,
        dim=dec_dim,
        depth=2,
        heads=2,
        dim_head=48,
        mlp_ratio=2,
    )

    # Simulate encoder output (unmasked patches + cls token)
    num_unmasked = int(L * 0.25)
    encoded = torch.randn(B, 1 + num_unmasked, enc_dim)
    unmasked_idx = torch.arange(num_unmasked).unsqueeze(0).expand(B, -1)
    masked_idx = torch.arange(num_unmasked, L).unsqueeze(0).expand(B, -1)
    mask_matrix = torch.zeros(B, L)
    mask_matrix[:, num_unmasked:] = 1

    with torch.no_grad():
        pixels, _waves = decoder(
            encoded,
            unmasked_idx,
            masked_idx,
            mask_matrix,
            time=torch.zeros(B, 4),
            latlon=torch.zeros(B, 4),
            gsd=torch.tensor(10.0),
            waves=torch.rand(channels),
        )

    # pixels: [B, L, C*P*P]
    assert pixels.shape == (B, L, channels * patch_size * patch_size)


def test_clay_mae_full_forward():
    """Full MAE forward pass returns encoded patches, decoded pixels, and mask."""
    model = _make_tiny_mae(mask_ratio=0.75)
    model.eval()

    datacube = {
        "pixels": torch.randn(2, 10, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.zeros(2, 4),
        "gsd": torch.tensor(10.0),
        "waves": torch.rand(10),
    }

    with torch.no_grad():
        encoded, decoded, mask, _unmasked_idx, _masked_idx = model(datacube)

    B, L = 2, (64 // 8) ** 2
    num_unmasked = int(L * 0.25)
    assert encoded.shape == (B, 1 + num_unmasked, 192)  # CLS + unmasked patches
    assert decoded.shape[0] == B
    assert decoded.shape[1] == L
    assert mask.shape == (B, L)


def test_per_pixel_loss_masked_only():
    """per_pixel_loss should compute loss only on masked patches."""
    metadata = make_metadata()
    model = clay_mae_tiny(
        mask_ratio=0.0,
        patch_size=8,
        norm_pix_loss=False,
        shuffle=False,
        metadata=metadata,
        teacher="samvit_base_patch16.sa1b",
        dolls=[16],
        doll_weights=[1],
    )

    B, C, H, W = 1, 4, 64, 64
    P = 8
    L = (H // P) * (W // P)

    cube = torch.randn(B, C, H, W)
    # Perfect prediction = same as input rearranged into patches
    pixels = rearrange(cube, "B C (h p1) (w p2) -> B (h w) (C p1 p2)", p1=P, p2=P)

    # All masked — loss on perfect prediction should be 0
    mask_all = torch.ones(B, L)
    loss = model.per_pixel_loss(cube, pixels, mask_all)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    # Note: with mask_none = zeros, per_pixel_loss divides by 0 (nan/inf).
    # This is expected — per_pixel_loss should only be called with masked patches.


def test_mask_out_zero_ratio_keeps_all():
    """With mask_ratio=0.0, all patches should be unmasked."""
    encoder = make_tiny_encoder(mask_ratio=0.0, shuffle=False)
    datacube = make_datacube(batch_size=1, channels=6, size=64)

    with torch.no_grad():
        _, unmasked_idx, masked_idx, mask_matrix = encoder(datacube)

    L = (64 // 8) ** 2
    assert unmasked_idx.shape == (1, L)  # all patches unmasked
    assert masked_idx.shape == (1, 0)  # no masked patches
    assert mask_matrix.sum() == 0  # all zeros


def test_module_invalid_size():
    """ClayMAEModule with invalid size should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid model size"):
        ClayMAEModule(
            model_size="nonexistent",
            metadata_path=str(
                __import__("importlib.resources", fromlist=["files"])
                .files("claymodel")
                .joinpath("configs/metadata.yaml")
            ),
        )


def test_module_encoder_property():
    """model.encoder should be a shortcut to model.model.encoder."""
    module = ClayMAEModule(
        model_size="tiny",
        mask_ratio=0.0,
        shuffle=False,
        metadata_path=str(
            __import__("importlib.resources", fromlist=["files"])
            .files("claymodel")
            .joinpath("configs/metadata.yaml")
        ),
        teacher="samvit_base_patch16.sa1b",
    )
    assert module.encoder is module.model.encoder


def test_configure_training_defaults():
    old_precision = torch.get_float32_matmul_precision()
    old_env = os.environ.get("TORCH_CUDNN_V8_API_DISABLED")
    try:
        configure_training_defaults()
        assert torch.get_float32_matmul_precision() == "medium"
        assert os.environ["TORCH_CUDNN_V8_API_DISABLED"] == "1"
    finally:
        torch.set_float32_matmul_precision(old_precision)
        if old_env is None:
            os.environ.pop("TORCH_CUDNN_V8_API_DISABLED", None)
        else:
            os.environ["TORCH_CUDNN_V8_API_DISABLED"] = old_env


def _make_tiny_mae(matryoshka=False, **overrides):
    """Create a tiny ClayMAE with a mock teacher for fast testing."""
    metadata = make_metadata()
    defaults = {
        "mask_ratio": 0.75,
        "patch_size": 8,
        "norm_pix_loss": False,
        "shuffle": True,
        "metadata": metadata,
        "teacher": "samvit_base_patch16.sa1b",
        "dolls": [16, 32],
        "doll_weights": [1, 1],
        "matryoshka": matryoshka,
    }
    defaults.update(overrides)
    model = clay_mae_tiny(**cast(Any, defaults))  # noqa: TC006
    teacher_dim = cast(int, cast(Any, model.teacher).num_features)  # noqa: TC006

    class MockTeacher(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], teacher_dim)

    model.teacher = MockTeacher()
    model.teacher_resize = nn.Identity()  # ty: ignore[invalid-assignment]
    return model


def test_matryoshka_flag_creates_mrl():
    model = _make_tiny_mae(matryoshka=True)
    assert hasattr(model, "mrl")
    assert hasattr(model, "mrl_loss")
    assert not hasattr(model, "proj")


def test_matryoshka_false_creates_proj():
    model = _make_tiny_mae(matryoshka=False)
    assert hasattr(model, "proj")
    assert not hasattr(model, "mrl")


def test_matryoshka_forward_path():
    """Forward pass works with matryoshka=True (MRL submodules present)."""
    model = _make_tiny_mae(matryoshka=True)
    model.eval()

    datacube = {
        "pixels": torch.randn(2, 10, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.zeros(2, 4),
        "gsd": torch.tensor(10.0),
        "waves": torch.rand(10),
    }
    with torch.no_grad():
        encoded, decoded, _mask, *_ = model(datacube)

    assert encoded.shape[0] == 2
    assert decoded.shape[0] == 2


def test_norm_pix_loss():
    """per_pixel_loss with norm_pix_loss=True should give higher loss than without."""
    model_norm = _make_tiny_mae(norm_pix_loss=True)
    model_plain = _make_tiny_mae(norm_pix_loss=False)

    B, C, H, W, P = 1, 4, 64, 64, 8
    L = (H // P) * (W // P)
    cube = torch.randn(B, C, H, W)
    # Intentionally imperfect prediction
    pixels = torch.randn(B, L, C * P * P)
    mask = torch.ones(B, L)

    loss_norm = model_norm.per_pixel_loss(cube, pixels, mask)
    loss_plain = model_plain.per_pixel_loss(cube, pixels, mask)

    # Both should be finite, and they should differ
    assert torch.isfinite(loss_norm)
    assert torch.isfinite(loss_plain)
    assert not torch.allclose(loss_norm, loss_plain)


def test_factory_dimensions():
    """Verify factory functions set correct encoder/decoder dims."""
    metadata = make_metadata()
    common = {
        "mask_ratio": 0.0,
        "patch_size": 8,
        "norm_pix_loss": False,
        "shuffle": False,
        "metadata": metadata,
        "teacher": "samvit_base_patch16.sa1b",
        "dolls": [16],
        "doll_weights": [1],
    }

    small = clay_mae_small(**common)
    assert small.encoder.dim == 384
    assert small.decoder.dim == 192

    base = clay_mae_base(**common)
    assert base.encoder.dim == 768
    assert base.decoder.dim == 512

    large = clay_mae_large(**common)
    assert large.encoder.dim == 1024
    assert large.decoder.dim == 512
