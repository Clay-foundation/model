"""Pydantic models for sensor metadata configuration."""

__all__ = ["BandStats", "PlatformMetadata", "Metadata"]

from pathlib import Path

import yaml
from pydantic import BaseModel, RootModel


class BandStats(BaseModel):
    mean: dict[str, float]
    std: dict[str, float]
    wavelength: dict[str, float]


class PlatformMetadata(BaseModel):
    band_order: list[str]
    rgb_indices: list[int] | None = None
    gsd: float
    bands: BandStats


class Metadata(RootModel[dict[str, PlatformMetadata]]):
    """Root container for sensor metadata.

    Supports dict-style access: metadata["sentinel-2-l2a"],
    "sensor" in metadata, metadata.keys(), iteration.
    """

    def __getitem__(self, key: str) -> PlatformMetadata:
        return self.root[key]

    def __len__(self) -> int:
        return len(self.root)

    def __contains__(self, key: object) -> bool:
        return key in self.root

    def __iter__(self):
        return iter(self.root)

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Metadata":
        """Load metadata from a YAML file."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
