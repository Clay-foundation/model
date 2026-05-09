"""Pydantic models for sensor metadata configuration."""

__all__ = ["BandStats", "PlatformMetadata", "load_metadata_yaml"]

from pathlib import Path

import yaml
from pydantic import BaseModel


class BandStats(BaseModel):
    mean: dict[str, float]
    std: dict[str, float]
    wavelength: dict[str, float]


class PlatformMetadata(BaseModel):
    band_order: list[str]
    rgb_indices: list[int] | None = None
    gsd: float
    bands: BandStats


def load_metadata_yaml(path: str | Path) -> dict[str, PlatformMetadata]:
    """Load and validate metadata from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    return {name: PlatformMetadata.model_validate(config) for name, config in raw.items()}
