"""Shared test fixtures for Clay model tests."""

from importlib.resources import files
from unittest.mock import MagicMock

import pytest
import torch

from claymodel.api import load_metadata
from claymodel.model import Encoder
from claymodel.module import ClayMAEModule


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False, help="Run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Need --slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@torch.no_grad()
def make_datacube(batch_size=2, channels=10, size=64):
    """Create a fake datacube matching the expected input format.

    Uses small 64x64 chips by default for fast tests (vs 256x256 in production).
    """
    return {
        "pixels": torch.randn(batch_size, channels, size, size),
        "time": torch.zeros(batch_size, 4),
        "latlon": torch.zeros(batch_size, 4),
        "gsd": torch.tensor(10.0),
        "waves": torch.rand(channels),
    }


def make_tiny_encoder(mask_ratio=0.0, shuffle=False):
    """Create a tiny encoder for fast testing (dim=192, random weights)."""
    return Encoder(
        mask_ratio=mask_ratio,
        patch_size=8,
        shuffle=shuffle,
        dim=192,
        depth=2,
        heads=4,
        dim_head=48,
        mlp_ratio=2,
    )


def make_metadata():
    """Load the bundled metadata for testing."""
    return load_metadata()


def _bundled_metadata_path():
    return str(files("claymodel").joinpath("configs/metadata.yaml"))


@pytest.fixture(scope="session")
def tiny_module():
    """Session-scoped tiny ClayMAEModule for integration tests.

    Uses the real teacher (downloaded and cached by timm), but encoder
    weights are random. The model is in eval mode with mask_ratio=0.
    """
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


@pytest.fixture(scope="session")
def tiny_training_module():
    """Session-scoped tiny ClayMAEModule for training integration tests.

    Uses the real teacher with masking enabled for training tests.
    """
    from torch import nn

    module = ClayMAEModule(
        model_size="tiny",
        mask_ratio=0.75,
        shuffle=True,
        metadata_path=_bundled_metadata_path(),
        teacher="samvit_base_patch16.sa1b",
    )

    # Mock teacher to avoid input size constraints in Tier 1
    # Tier 2 (slow) tests use the real teacher via tiny_real_teacher_module
    teacher_dim = module.model.teacher.num_features

    class MockTeacher(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], teacher_dim)

    module.model.teacher = MockTeacher()
    module.model.teacher_resize = nn.Identity()
    module.log = MagicMock()
    return module


@pytest.fixture(scope="session")
def tiny_real_teacher_module():
    """Session-scoped module with REAL teacher for slow integration tests.

    The SAM teacher (samvit_base_patch16.sa1b) expects input divisible by 16.
    We override teacher_resize to 512x512 (512/16=32) since the default 518
    was designed for DINOv2 (518/14=37).
    """
    from torchvision.transforms import v2

    module = ClayMAEModule(
        model_size="tiny",
        mask_ratio=0.75,
        shuffle=True,
        metadata_path=_bundled_metadata_path(),
        teacher="samvit_base_patch16.sa1b",
    )
    module.model.teacher_resize = v2.Resize(size=(512, 512))
    module.log = MagicMock()
    return module


@pytest.fixture(scope="session")
def metadata():
    """Session-scoped metadata."""
    return load_metadata()
