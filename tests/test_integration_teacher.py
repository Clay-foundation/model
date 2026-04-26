"""Tier 2 integration tests: real teacher model, @pytest.mark.slow.

Run with: pytest tests/ --slow
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import make_metadata


@pytest.mark.slow
def test_training_step_real_teacher_sentinel2(tiny_real_teacher_module):
    """Full training step with real teacher produces finite loss."""
    module = tiny_real_teacher_module
    module.log = MagicMock()

    batch = {
        "pixels": torch.randn(2, 10, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.zeros(2, 4),
        "platform": ["sentinel-2-l2a", "sentinel-2-l2a"],
    }
    loss = module.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert module.log.call_count == 6


@pytest.mark.slow
def test_validation_step_real_teacher_sentinel2(tiny_real_teacher_module):
    """Validation step with real teacher produces finite loss."""
    module = tiny_real_teacher_module
    module.log = MagicMock()

    batch = {
        "pixels": torch.randn(2, 10, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.zeros(2, 4),
        "platform": ["sentinel-2-l2a", "sentinel-2-l2a"],
    }
    loss = module.validation_step(batch, batch_idx=0)

    assert torch.isfinite(loss)
    assert loss.item() > 0


@pytest.mark.slow
def test_training_step_real_teacher_sentinel1_sar(tiny_real_teacher_module):
    """SAR synthetic RGB feeds correctly into real teacher."""
    module = tiny_real_teacher_module
    module.log = MagicMock()

    metadata = make_metadata()
    n_bands = len(metadata["sentinel-1-rtc"].bands.wavelength)

    batch = {
        "pixels": torch.randn(2, n_bands, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.zeros(2, 4),
        "platform": ["sentinel-1-rtc", "sentinel-1-rtc"],
    }
    loss = module.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)


@pytest.mark.slow
def test_training_step_matryoshka_real_teacher():
    """MRL loss works with real teacher output."""
    from torchvision.transforms import v2

    from tests.conftest import _bundled_metadata_path

    module = __import__("claymodel.module", fromlist=["ClayMAEModule"]).ClayMAEModule(
        model_size="tiny",
        mask_ratio=0.75,
        shuffle=True,
        metadata_path=_bundled_metadata_path(),
        teacher="samvit_base_patch16.sa1b",
        matryoshka=True,
        dolls=[16, 32, 64, 128],
        doll_weights=[1, 1, 1, 1],
    )
    module.model.teacher_resize = v2.Resize(size=(512, 512))
    module.log = MagicMock()

    batch = {
        "pixels": torch.randn(2, 10, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.zeros(2, 4),
        "platform": ["sentinel-2-l2a", "sentinel-2-l2a"],
    }
    loss = module.training_step(batch, batch_idx=0)

    assert torch.isfinite(loss)


@pytest.mark.slow
def test_teacher_is_frozen_after_training_step(tiny_real_teacher_module):
    """Teacher stays frozen through training."""
    module = tiny_real_teacher_module
    module.log = MagicMock()

    module.on_train_epoch_start()

    batch = {
        "pixels": torch.randn(2, 10, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.zeros(2, 4),
        "platform": ["sentinel-2-l2a", "sentinel-2-l2a"],
    }
    module.training_step(batch, batch_idx=0)

    for param in module.model.teacher.parameters():
        assert not param.requires_grad
    assert not module.model.teacher.training


@pytest.mark.slow
def test_representation_loss_similar_vs_random(tiny_real_teacher_module):
    """Representation loss is lower when projection output matches target."""
    module = tiny_real_teacher_module
    pixels = torch.randn(1, 10, 64, 64)
    encoder_dim = module.model.encoder.dim  # 192 for tiny

    # Get real teacher target (256-dim for SAM)
    target = module._teacher_target(pixels, "sentinel-2-l2a")

    # Project two different CLS tokens and compare losses
    # One that the projection maps close to target, one random
    with torch.no_grad():
        # Find a CLS token whose projection is close to target
        # by inverting: cls = proj.weight^T @ target (approximate)
        proj = module.model.proj
        cls_good = F.linear(target, proj.weight.T)  # [1, encoder_dim]
        cls_random = torch.randn(1, encoder_dim)

    loss_good = module._representation_loss(cls_good, target)
    loss_random = module._representation_loss(cls_random, target)

    assert loss_good < loss_random


@pytest.mark.slow
def test_full_mae_forward_decode_reconstruct(tiny_real_teacher_module):
    """Full encode-decode chain without mocking produces finite outputs."""
    model = tiny_real_teacher_module.model
    metadata = make_metadata()
    platform = "sentinel-2-l2a"
    n_bands = len(metadata[platform].bands.wavelength)

    datacube = {
        "pixels": torch.randn(2, n_bands, 64, 64),
        "time": torch.zeros(2, 4),
        "latlon": torch.zeros(2, 4),
        "gsd": torch.tensor(metadata[platform].gsd),
        "waves": torch.tensor(list(metadata[platform].bands.wavelength.values())),
    }

    with torch.no_grad():
        encoded, decoded, mask, unmasked_idx, masked_idx = model(datacube)

    assert torch.isfinite(encoded).all()
    assert torch.isfinite(decoded).all()

    # per_pixel_loss should be finite on the decoded output
    loss = model.per_pixel_loss(datacube["pixels"], decoded, mask)
    assert torch.isfinite(loss)
