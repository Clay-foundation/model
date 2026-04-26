"""
Clay Foundation Model - An open source AI model and interface for Earth observation.

This package provides tools for working with satellite imagery and Earth
observation data using foundation models built on Vision Transformers.

Example usage:
    from claymodel import ClayMAEModule, load_metadata

    metadata = load_metadata()
    model = ClayMAEModule(model_size="large")
"""

from importlib.metadata import version

from claymodel.api import embed, load_metadata, load_model, normalize
from claymodel.metadata import Metadata
from claymodel.model import clay_mae_base, clay_mae_large, clay_mae_small, clay_mae_tiny
from claymodel.module import ClayMAEModule

__version__: str = version("claymodel")


__all__ = [
    "ClayMAEModule",
    "clay_mae_base",
    "clay_mae_large",
    "clay_mae_small",
    "clay_mae_tiny",
    "Metadata",
    "load_metadata",
    "load_model",
    "embed",
    "normalize",
]
