"""Test ClayMAEModule training orchestration."""

import random
from importlib.resources import files
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from claymodel.module import ClayMAEModule
from tests.conftest import make_metadata


def _bundled_metadata_path():
    return str(files("claymodel").joinpath("configs/metadata.yaml"))


def _make_module(**overrides: object):
    defaults = {
        "model_size": "tiny",
        "mask_ratio": 0.75,
        "shuffle": True,
        "metadata_path": _bundled_metadata_path(),
        "teacher": "samvit_base_patch16.sa1b",
    }
    defaults.update(overrides)
    module = ClayMAEModule(**cast(Any, defaults))  # noqa: TC006

    # Mock the teacher to avoid input size constraints
    teacher_dim = module.teacher.num_features

    class MockTeacher(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], cast(int, teacher_dim))  # noqa: TC006

    module.teacher = MockTeacher()
    module.teacher_resize = nn.Identity()  # ty: ignore[invalid-assignment]
    return module


def _make_batch(platform="sentinel-2-l2a", n_bands=10, latlon_val=0.0):
    return {
        "pixels": torch.randn(2, n_bands, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.full((2, 4), latlon_val),
        "platform": [platform, platform],
    }


def test_configure_optimizers_structure():
    module = _make_module()
    result = module.configure_optimizers()

    assert "optimizer" in result
    assert "lr_scheduler" in result
    assert result["lr_scheduler"]["interval"] == "step"


def test_configure_optimizers_uses_hparams():
    module = _make_module(lr=1e-3, wd=0.1, b1=0.85, b2=0.9)
    result = module.configure_optimizers()
    optimizer = result["optimizer"]

    pg = optimizer.param_groups[0]
    assert abs(pg["lr"] - 1e-3) < 1e-9
    assert abs(pg["weight_decay"] - 0.1) < 1e-9
    assert pg["betas"] == (0.85, 0.9)


def test_on_train_epoch_start_sets_teacher_eval():
    module = _make_module()
    module.teacher.train()
    assert module.teacher.training

    module.on_train_epoch_start()
    assert not module.teacher.training


def test_training_step_returns_finite_loss():
    module = _make_module()
    module.log = MagicMock()

    batch = _make_batch()
    loss = module.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)


def test_training_step_logs_train_phase():
    module = _make_module()
    module.log = MagicMock()

    batch = _make_batch()
    module.training_step(batch, batch_idx=0)

    names = [c.kwargs["name"] for c in module.log.call_args_list]
    assert all("train" in n for n in names)
    assert module.log.call_count == 6


def test_validation_step_returns_finite_loss():
    module = _make_module()
    module.log = MagicMock()

    batch = _make_batch()
    loss = module.validation_step(batch, batch_idx=0)

    assert torch.isfinite(loss)


def test_validation_step_logs_val_phase():
    module = _make_module()
    module.log = MagicMock()

    batch = _make_batch()
    module.validation_step(batch, batch_idx=0)

    names = [c.kwargs["name"] for c in module.log.call_args_list]
    assert all("val" in n for n in names)


def test_channel_dropout_with_nonzero_latlon():
    """Channel dropout should trigger when latlon is nonzero."""
    module = _make_module()
    pixels = torch.randn(2, 10, 64, 64)
    latlon = torch.ones(2, 4)

    # Find a seed that triggers the drop-all branch
    for seed in range(100):
        random.seed(seed)
        if random.random() < 0.10:
            random.seed(seed)
            break

    dropped = module._channel_dropout(pixels, latlon)
    # At least one sample should have been modified
    assert not torch.equal(pixels, dropped)


def test_channel_dropout_skips_zero_latlon():
    """Channel dropout should not modify samples with zero latlon."""
    module = _make_module()
    pixels = torch.randn(2, 10, 64, 64)
    latlon = torch.zeros(2, 4)

    random.seed(0)
    dropped = module._channel_dropout(pixels, latlon)
    assert torch.equal(pixels, dropped)


def test_modis_platform_scaling():
    """MODIS platform should scale reconstruction loss by /10."""
    metadata = make_metadata()
    if "modis" not in metadata:
        pytest.skip("MODIS not in metadata")

    module = _make_module()
    module.log = MagicMock()

    n_bands = len(metadata["modis"].bands.wavelength)
    batch = _make_batch(platform="modis", n_bands=n_bands)
    loss = module.validation_step(batch, batch_idx=0)

    assert torch.isfinite(loss)


def test_sentinel1_rgb_construction():
    """Sentinel-1 should construct synthetic RGB for teacher."""
    metadata = make_metadata()
    n_bands = len(metadata["sentinel-1-rtc"].bands.wavelength)

    module = _make_module()
    module.log = MagicMock()

    batch = _make_batch(platform="sentinel-1-rtc", n_bands=n_bands)
    loss = module.validation_step(batch, batch_idx=0)

    assert torch.isfinite(loss)


def test_matryoshka_training_step():
    """Training step works with matryoshka=True."""
    module = _make_module(matryoshka=True, dolls=[16, 32, 64, 128], doll_weights=[1, 1, 1, 1])
    module.log = MagicMock()

    batch = _make_batch()
    loss = module.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)
