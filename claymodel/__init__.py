"""
Clay Foundation Model - An open source AI model and interface for Earth observation.

This package provides tools for working with satellite imagery and Earth
observation data using foundation models built on Vision Transformers.

Example usage:
    from claymodel import ClayMAEModule, load_metadata

    metadata = load_metadata()
    model = ClayMAEModule(model_size="large")
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("claymodel")
except PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name):
    """Lazy imports to avoid pulling in torch at module import time."""
    if name == "ClayMAEModule":
        from claymodel.module import ClayMAEModule

        return ClayMAEModule
    if name == "ClayDataModule":
        from claymodel.datamodule import ClayDataModule

        return ClayDataModule
    if name in ("clay_mae_base", "clay_mae_large", "clay_mae_small", "clay_mae_tiny"):
        from claymodel import model as _model

        return getattr(_model, name)
    if name == "load_metadata":
        from claymodel.api import load_metadata

        return load_metadata
    if name == "load_model":
        from claymodel.api import load_model

        return load_model
    if name == "embed":
        from claymodel.api import embed

        return embed
    if name == "normalize":
        from claymodel.api import normalize

        return normalize
    raise AttributeError(f"module 'claymodel' has no attribute {name!r}")


__all__ = [
    "ClayMAEModule",
    "ClayDataModule",
    "clay_mae_base",
    "clay_mae_large",
    "clay_mae_small",
    "clay_mae_tiny",
    "load_metadata",
    "load_model",
    "embed",
    "normalize",
]
