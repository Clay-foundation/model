"""High-level Clay model API."""

__all__ = ["EmbeddingResult", "embed", "load_metadata", "load_model", "normalize"]

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from claymodel.metadata import PlatformMetadata, load_metadata_yaml
from claymodel.module import ClayMAEModule


def load_metadata(
    path: str | Path | None = None,
) -> dict[str, PlatformMetadata]:
    """Load sensor metadata from YAML."""
    if path is None:
        path = str(files("claymodel").joinpath("configs/metadata.yaml"))
    return load_metadata_yaml(path)


def normalize(
    pixels: torch.Tensor,
    sensor: str,
    metadata: dict[str, PlatformMetadata] | None = None,
) -> torch.Tensor:
    """Normalize pixel values with sensor-specific statistics."""
    if metadata is None:
        metadata = load_metadata()

    if sensor not in metadata:
        raise ValueError(f"Unknown sensor {sensor!r}. Available: {list(metadata.keys())}")

    sensor_meta = metadata[sensor]
    mean = torch.tensor(list(sensor_meta.bands.mean.values()), dtype=pixels.dtype).view(1, -1, 1, 1)
    std = torch.tensor(list(sensor_meta.bands.std.values()), dtype=pixels.dtype).view(1, -1, 1, 1)

    mean = mean.to(pixels.device)
    std = std.to(pixels.device)

    return (pixels - mean) / std


def load_model(
    size: str = "large",
    ckpt_path: str | None = None,
    device: str = "cpu",
    metadata_path: str | Path | None = None,
) -> ClayMAEModule:
    """Load a Clay MAE model ready for inference.

    Creates a ClayMAEModule and optionally loads weights from a checkpoint.
    The model is returned in eval mode with mask_ratio=0 and shuffle=False
    for deterministic inference.

    Note: The model includes a DINOv2 teacher (~300MB) that is downloaded
    on first use. The teacher is frozen and not needed for embedding
    extraction, but is part of the architecture.

    Args:
        size: Model size - "tiny", "small", "base", or "large".
        ckpt_path: Path to checkpoint file. If None, creates model with
            random weights (useful for testing).
        device: Device to load model onto ("cpu", "cuda", etc.).
        metadata_path: Path to a custom metadata YAML file. If None,
            uses the bundled metadata with common public sensors.

    Returns:
        ClayMAEModule instance in eval mode.

    Example:
        >>> model = load_model("large", ckpt_path="clay-v1.5.ckpt")
        >>> model = load_model("large", metadata_path="my_sensors.yaml")
    """
    resolved_path = (
        str(metadata_path)
        if metadata_path
        else str(files("claymodel").joinpath("configs/metadata.yaml"))
    )

    if ckpt_path is not None:
        model = ClayMAEModule.load_from_checkpoint(
            ckpt_path,
            metadata_path=resolved_path,
            map_location=device,
        )
    else:
        model = ClayMAEModule(
            model_size=size,
            mask_ratio=0.0,
            shuffle=False,
            metadata_path=resolved_path,
        )

    model.model.encoder.mask_ratio = 0.0
    model.model.encoder.shuffle = False
    model.eval()
    return model.to(device)


@dataclass
class EmbeddingResult:
    """Clay embeddings plus export metadata."""

    embeddings: torch.Tensor
    sensor: str = ""
    gsd: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> torch.Size:
        return self.embeddings.shape


def embed(  # noqa: PLR0913
    input_data: torch.Tensor | np.ndarray,
    sensor: str,
    model: ClayMAEModule | None = None,
    ckpt_path: str | None = None,
    device: str = "cpu",
    time: torch.Tensor | None = None,
    latlon: torch.Tensor | None = None,
    metadata: dict[str, PlatformMetadata] | None = None,
) -> EmbeddingResult:
    """Embed pixels with a Clay model."""
    if metadata is None:
        metadata = load_metadata()

    if isinstance(input_data, np.ndarray):
        pixels = torch.from_numpy(np.asarray(input_data, dtype=np.float32))
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
    elif isinstance(input_data, torch.Tensor):
        pixels = input_data.float()
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
    else:
        raise TypeError(f"input_data must be a Tensor or ndarray, got {type(input_data)}")

    if sensor not in metadata:
        raise ValueError(f"Unknown sensor {sensor!r}. Available: {list(metadata.keys())}")

    sensor_meta = metadata[sensor]
    pixels = pixels.to(device)

    if sensor == "sentinel-1-rtc":
        pixels = pixels.clamp(min=1e-10)
        pixels = 10 * torch.log10(pixels)

    pixels = normalize(pixels, sensor, metadata=metadata)

    B = pixels.shape[0]
    waves = torch.tensor(list(sensor_meta.bands.wavelength.values()), device=device)
    gsd = torch.tensor(sensor_meta.gsd, dtype=torch.float32, device=device)

    if time is None:
        time = torch.zeros(B, 4, device=device)
    if latlon is None:
        latlon = torch.zeros(B, 4, device=device)

    datacube = {
        "pixels": pixels,
        "time": time,
        "latlon": latlon,
        "gsd": gsd,
        "waves": waves,
    }

    if model is None:
        if ckpt_path is None:
            raise ValueError("Either model or ckpt_path must be provided")
        model = load_model(ckpt_path=ckpt_path, device=device)

    with torch.no_grad():
        encoded, *_ = model.encoder(datacube)
        cls_embeddings = encoded[:, 0, :]

    return EmbeddingResult(
        embeddings=cls_embeddings,
        sensor=sensor,
        gsd=float(sensor_meta.gsd),
        metadata={"latlon": latlon, "time": time},
    )
