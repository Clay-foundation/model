"""Clay model package exports."""

from importlib.metadata import version

from claymodel.api import EmbeddingResult, embed, load_metadata, load_model, normalize
from claymodel.metadata import PlatformMetadata
from claymodel.model import Encoder, clay_mae_base, clay_mae_large, clay_mae_small, clay_mae_tiny

__version__: str = version("claymodel")


__all__ = [
    "EmbeddingResult",
    "Encoder",
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
