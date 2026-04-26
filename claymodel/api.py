"""
High-level API for Clay Foundation Model.

Provides convenience functions for loading metadata, models, normalizing
inputs, and computing embeddings.
"""

from __future__ import annotations

__all__ = ["load_metadata", "normalize", "load_model", "EmbeddingResult", "embed"]

import warnings
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from claymodel.inference.elle import ELLEProbe
from claymodel.metadata import PlatformMetadata, load_metadata_yaml
from claymodel.module import ClayMAEModule


def load_metadata(
    path: str | Path | None = None,
) -> dict[str, PlatformMetadata]:
    """Load sensor metadata from a YAML file.

    Args:
        path: Path to a metadata YAML file. If None, loads the bundled
            metadata with common public sensors (Sentinel-2, Sentinel-1,
            NAIP, Landsat, MODIS, etc.).

    Returns a validated dict mapping sensor names to PlatformMetadata.

    Example:
        >>> from claymodel import load_metadata
        >>> metadata = load_metadata()  # bundled defaults
        >>> metadata = load_metadata("my_sensors.yaml")  # custom sensors
        >>> metadata["sentinel-2-l2a"].gsd
        10
    """
    if path is None:
        path = str(files("claymodel").joinpath("configs/metadata.yaml"))
    return load_metadata_yaml(path)


def _bundled_metadata_path() -> str:
    """Resolve the absolute path to the bundled metadata.yaml."""
    return str(files("claymodel").joinpath("configs/metadata.yaml"))


def normalize(
    pixels: torch.Tensor,
    sensor: str,
    metadata: dict[str, PlatformMetadata] | None = None,
) -> torch.Tensor:
    """Normalize raw pixel values using sensor-specific z-score statistics.

    Applies the same normalization used during Clay v1.5 training:
    per-band z-score using mean/std from metadata.yaml.

    For Sentinel-1 SAR data, pixels should already be in dB scale
    (10 * log10(linear_power)). The datamodule does this conversion
    automatically, but when using this function directly you must
    convert SAR data to dB first.

    Args:
        pixels: [B, C, H, W] tensor of raw pixel values.
        sensor: Sensor name matching a key in metadata.yaml
            (e.g., "sentinel-2-l2a", "sentinel-1-rtc", "naip").
        metadata: Optional pre-loaded metadata Box. If None, loads
            the bundled metadata.

    Returns:
        [B, C, H, W] normalized tensor.
    """
    if metadata is None:
        metadata = load_metadata()

    if sensor not in metadata:
        raise ValueError(
            f"Unknown sensor {sensor!r}. Available: {list(metadata.keys())}"
        )

    sensor_meta = metadata[sensor]
    mean = torch.tensor(list(sensor_meta.bands.mean.values()), dtype=pixels.dtype).view(
        1, -1, 1, 1
    )
    std = torch.tensor(list(sensor_meta.bands.std.values()), dtype=pixels.dtype).view(
        1, -1, 1, 1
    )

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
    resolved_path = str(metadata_path) if metadata_path else _bundled_metadata_path()

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

    # Ensure inference-ready settings
    model.model.encoder.mask_ratio = 0.0
    model.model.encoder.shuffle = False
    model.eval()
    return model.to(device)


@dataclass
class EmbeddingResult:
    """Container for Clay model embeddings with export capabilities.

    Attributes:
        embeddings: [N, D] tensor of embeddings (D=1024 for large model).
        sensor: Sensor name used for the input.
        gsd: Ground sampling distance.
        metadata: Additional metadata dict (coordinates, timestamps, etc.).
    """

    embeddings: torch.Tensor
    sensor: str = ""
    gsd: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> torch.Size:
        return self.embeddings.shape

    def to_geoparquet(self, path: str | Path) -> object:
        """Export embeddings to GeoParquet format.

        Requires the [pipeline] extras: pip install claymodel[pipeline]

        Args:
            path: Output file path (should end in .parquet or .geoparquet).
        """
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import Point
        except ImportError as e:
            raise ImportError(
                "GeoParquet export requires geopandas. "
                "Install with: pip install claymodel[pipeline]"
            ) from e

        records = []
        for i in range(self.embeddings.shape[0]):
            record = {
                "embedding": self.embeddings[i].cpu().numpy().tolist(),
                "sensor": self.sensor,
                "gsd": self.gsd,
            }
            # Add coordinates if available
            if "latlon" in self.metadata and self.metadata["latlon"] is not None:
                lat = self.metadata["latlon"][i][0].item()
                lon = self.metadata["latlon"][i][1].item()
                record["geometry"] = Point(lon, lat)
            else:
                record["geometry"] = None
            records.append(record)

        df = pd.DataFrame(records)
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        gdf.to_parquet(path)
        return gdf


def embed(  # noqa: PLR0913
    input_data: torch.Tensor | np.ndarray | str | Path,
    sensor: str,
    model: ClayMAEModule | None = None,
    ckpt_path: str | None = None,
    device: str = "cpu",
    time: torch.Tensor | None = None,
    latlon: torch.Tensor | None = None,
    quality: bool = False,
    metadata: dict[str, PlatformMetadata] | None = None,
) -> EmbeddingResult:
    """One-line embedding API for Clay Foundation Model.

    Accepts raw pixel data (as a tensor, numpy array, or GeoTIFF path),
    normalizes it using sensor-specific statistics, runs the encoder,
    and returns embeddings. For Sentinel-1 SAR, pass raw linear power
    values — the function converts to dB internally before normalization.

    Args:
        input_data: One of:
            - torch.Tensor of shape [B, C, H, W] (raw pixel values)
            - numpy.ndarray of shape [B, C, H, W] or [C, H, W]
            - str/Path to a GeoTIFF file (requires rasterio from [cli] extras)
        sensor: Sensor name matching metadata (e.g., "sentinel-2-l2a").
        model: Pre-loaded ClayMAEModule. If None, loads from ckpt_path.
        ckpt_path: Path to checkpoint (used if model is None).
        device: Device for computation.
        time: Optional [B, 4] tensor of (week_sin, week_cos, hour_sin, hour_cos).
            Defaults to zeros (unknown time).
        latlon: Optional [B, 4] tensor of (lat_sin, lat_cos, lon_sin, lon_cos).
            Defaults to zeros (unknown location).
        quality: If True, compute ELLE quality score (requires trained probe).
        metadata: Pre-loaded metadata Box. If None, loads bundled defaults.
            Use load_metadata("my_sensors.yaml") to load custom sensors.

    Returns:
        EmbeddingResult with .embeddings tensor of shape [N, D] and
        .to_geoparquet() method. D=1024 for the large model.

    Example:
        >>> from claymodel import embed, load_metadata
        >>> result = embed(pixels, sensor="sentinel-2-l2a", ckpt_path="v1.5.ckpt")
        >>> metadata = load_metadata("my_sensors.yaml")
        >>> result = embed(pixels, sensor="my-sensor", model=model, metadata=metadata)
    """
    if metadata is None:
        metadata = load_metadata()

    # Handle GeoTIFF input
    if isinstance(input_data, (str, Path)):
        input_data = str(input_data)
        try:
            import rasterio
        except ImportError as e:
            raise ImportError(
                "GeoTIFF reading requires rasterio. "
                "Install with: pip install claymodel[cli]"
            ) from e

        with rasterio.open(input_data) as src:
            pixels = src.read().astype(np.float32)  # [C, H, W]
            pixels = torch.from_numpy(pixels).unsqueeze(0)  # [1, C, H, W]
    elif isinstance(input_data, np.ndarray):
        pixels = torch.from_numpy(input_data.astype(np.float32))
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    elif isinstance(input_data, torch.Tensor):
        pixels = input_data.float()
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
    else:
        raise TypeError(
            f"input_data must be a Tensor, ndarray, or file path, "
            f"got {type(input_data)}"
        )

    if sensor not in metadata:
        raise ValueError(
            f"Unknown sensor {sensor!r}. Available: {list(metadata.keys())}"
        )

    sensor_meta = metadata[sensor]
    pixels = pixels.to(device)

    # Sentinel-1 SAR: convert linear power to dB scale.
    # If your SAR data is already in dB, skip this by normalizing manually:
    #   normalized = normalize(db_pixels, "sentinel-1-rtc")
    # and passing the result directly as a datacube to model.encoder().
    if sensor == "sentinel-1-rtc":
        pixels = pixels.clamp(min=1e-10)
        pixels = 10 * torch.log10(pixels)

    # Normalize
    pixels = normalize(pixels, sensor, metadata=metadata)

    # Build datacube
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

    # Load or use provided model
    if model is None:
        if ckpt_path is None:
            raise ValueError("Either model or ckpt_path must be provided")
        model = load_model(ckpt_path=ckpt_path, device=device)

    # Run encoder
    with torch.no_grad():
        encoded, *_ = model.encoder(datacube)
        cls_embeddings = encoded[:, 0, :]  # [B, D]

    result = EmbeddingResult(
        embeddings=cls_embeddings,
        sensor=sensor,
        gsd=float(sensor_meta.gsd),
        metadata={"latlon": latlon, "time": time},
    )

    # Optional ELLE quality scoring
    if quality:
        try:
            probe = ELLEProbe.default()
            result.metadata["quality_score"] = probe.score(cls_embeddings)
        except FileNotFoundError:
            warnings.warn(
                "ELLE probe not available. Install or provide probe weights.",
                stacklevel=2,
            )

    return result
