"""Clay model package exports."""

from importlib.metadata import version

from claymodel.api import EmbeddingResult, embed, load_metadata, load_model, normalize
from claymodel.metadata import PlatformMetadata
from claymodel.model import Encoder

__version__: str = version("claymodel")


__all__ = [
    "EmbeddingResult",
    "Encoder",
    "PlatformMetadata",
    "embed",
    "load_metadata",
    "load_model",
    "normalize",
]
