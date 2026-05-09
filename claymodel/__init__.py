"""Clay model package exports."""

from importlib.metadata import version

from claymodel.api import embed, load_metadata, load_model, normalize
from claymodel.metadata import PlatformMetadata
from claymodel.model import clay_mae_base, clay_mae_large, clay_mae_small, clay_mae_tiny
from claymodel.module import ClayMAEModule

__version__: str = version("claymodel")


__all__ = [
    "ClayMAEModule",
    "PlatformMetadata",
    "clay_mae_base",
    "clay_mae_large",
    "clay_mae_small",
    "clay_mae_tiny",
    "embed",
    "load_metadata",
    "load_model",
    "normalize",
]
