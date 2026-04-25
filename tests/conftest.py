"""Shared test fixtures for Clay model tests."""

import torch
import yaml
from box import Box

from claymodel.model import Encoder


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
    from importlib.resources import files  # noqa: PLC0415

    metadata_file = files("claymodel").joinpath("configs/metadata.yaml")
    return Box(yaml.safe_load(metadata_file.read_text()))
