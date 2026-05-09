"""High-level Clay model API."""

__all__ = ["EmbeddingResult", "embed", "load_metadata", "load_model", "normalize"]

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from claymodel.metadata import PlatformMetadata, load_metadata_yaml
from claymodel.model import Encoder
from claymodel.utils import load_encoder_weights

_ENCODER_CONFIGS: dict[str, dict[str, int]] = {
    "tiny": {"dim": 192, "depth": 6, "heads": 4, "dim_head": 48, "mlp_ratio": 2},
    "small": {"dim": 384, "depth": 6, "heads": 6, "dim_head": 64, "mlp_ratio": 2},
    "base": {"dim": 768, "depth": 12, "heads": 12, "dim_head": 64, "mlp_ratio": 4},
    "large": {"dim": 1024, "depth": 24, "heads": 16, "dim_head": 64, "mlp_ratio": 4},
}


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
) -> Encoder:
    """Load a Clay encoder ready for inference.

    Creates an Encoder and optionally loads weights from a checkpoint.
    The encoder is returned in eval mode with mask_ratio=0 and shuffle=False
    for deterministic inference. No teacher model is downloaded.

    Args:
        size: Model size - "tiny", "small", "base", or "large".
        ckpt_path: Path to checkpoint file. If None, creates encoder with
            random weights (useful for testing).
        device: Device to load model onto ("cpu", "cuda", etc.).

    Returns:
        Encoder instance in eval mode.

    Example:
        >>> encoder = load_model("large", ckpt_path="clay-v1.5.ckpt")
    """
    if size not in _ENCODER_CONFIGS:
        raise ValueError(f"Invalid size {size!r}. Expected one of {list(_ENCODER_CONFIGS.keys())}")

    encoder = Encoder(
        mask_ratio=0.0,
        patch_size=8,
        shuffle=False,
        **_ENCODER_CONFIGS[size],
    )

    if ckpt_path is not None:
        load_encoder_weights(encoder, ckpt_path, device=device, freeze=False)

    encoder.eval()
    return encoder.to(device)


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
    model: Encoder | None = None,
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
    expected_bands = len(sensor_meta.band_order)
    actual_bands = pixels.shape[1]
    if actual_bands != expected_bands:
        raise ValueError(
            f"Expected {expected_bands} bands for {sensor!r}, got {actual_bands}. "
            f"Band order: {sensor_meta.band_order}"
        )

    _, _, H, W = pixels.shape
    patch_size = 8
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Spatial dims must be divisible by {patch_size}, got ({H}, {W})")

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
        encoded, *_ = model(datacube)
        cls_embeddings = encoded[:, 0, :]

    return EmbeddingResult(
        embeddings=cls_embeddings,
        sensor=sensor,
        gsd=float(sensor_meta.gsd),
        metadata={"latlon": latlon, "time": time},
    )
