"""Sentinel-2 inference pipeline for Clay Foundation Model.

Provides functions to search Earth Search STAC, load Sentinel-2 COGs via
lazycogs, chip into model-ready patches, and generate embeddings with
spatial metadata. Usable both programmatically and via the CLI.
"""

__all__ = [
    "EmbeddingChip",
    "SearchResult",
    "chip_array",
    "encode_latlon",
    "encode_timestamp",
    "generate_embeddings",
    "load_scene",
    "search_scene",
]

import asyncio
import math
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import lazycogs
import numpy as np
import rustac
import torch
from shapely.geometry import box

from claymodel.api import load_metadata, load_model, normalize
from claymodel.metadata import PlatformMetadata
from claymodel.model import Encoder

# Earth Search STAC API endpoint
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

# Sentinel-2 bands in Clay metadata order
S2_BANDS = [
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir16",
    "swir22",
]

SENSOR = "sentinel-2-l2a"


def encode_timestamp(dt: datetime) -> torch.Tensor:
    """Encode a datetime as cyclic (sin, cos) features for week and hour.

    Returns a [4] tensor: [week_sin, week_cos, hour_sin, hour_cos].

    Args:
        dt: Datetime to encode.

    Returns:
        Tensor of shape [4] with cyclic temporal encoding.
    """
    week = dt.isocalendar()[1] * 2 * math.pi / 52
    hour = dt.hour * 2 * math.pi / 24
    return torch.tensor(
        [math.sin(week), math.cos(week), math.sin(hour), math.cos(hour)],
        dtype=torch.float32,
    )


def encode_latlon(lat: float, lon: float) -> torch.Tensor:
    """Encode lat/lon as cyclic (sin, cos) features.

    Returns a [4] tensor: [lat_sin, lat_cos, lon_sin, lon_cos].

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        Tensor of shape [4] with cyclic spatial encoding.
    """
    lat_rad = lat * math.pi / 180
    lon_rad = lon * math.pi / 180
    return torch.tensor(
        [math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)],
        dtype=torch.float32,
    )


@dataclass
class SearchResult:
    """Result of a STAC search for a Sentinel-2 scene."""

    scene_id: str
    parquet_path: Path
    datetime: datetime
    bbox: list[float]
    epsg: int
    cloud_cover: float | None = None
    properties: dict = field(default_factory=dict)


@dataclass
class EmbeddingChip:
    """A single chip's embedding with spatial metadata."""

    embedding: torch.Tensor  # [D] CLS embedding
    row: int  # chip row index in the grid
    col: int  # chip column index in the grid
    bbox: tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in scene CRS
    epsg: int
    scene_id: str
    datetime: datetime


async def _search_scene_async(
    scene_id: str,
    parquet_path: Path,
) -> int:
    """Search Earth Search for a specific scene ID and save to parquet.

    Args:
        scene_id: Sentinel-2 scene ID (e.g. S2A_MSIL2A_20240101T...).
        parquet_path: Path to write the STAC items parquet file.

    Returns:
        Number of items found.
    """
    return await rustac.search_to(
        str(parquet_path),
        EARTH_SEARCH_URL,
        collections=[SENSOR],
        ids=[scene_id],
        max_items=1,
    )


def search_scene(
    scene_id: str,
    output_dir: str | Path | None = None,
) -> SearchResult:
    """Search Earth Search STAC for a Sentinel-2 scene by ID.

    Args:
        scene_id: Sentinel-2 scene ID.
        output_dir: Directory to write the parquet file. Uses a temp dir if None.

    Returns:
        SearchResult with scene metadata and path to parquet file.

    Raises:
        ValueError: If the scene is not found.
        ImportError: If rustac is not installed.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="clay_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "items.parquet"

    try:
        count = asyncio.run(_search_scene_async(scene_id, parquet_path))
    except Exception as exc:
        if "no items" in str(exc).lower():
            raise ValueError(
                f"Scene {scene_id!r} not found in Earth Search. "
                "Verify the scene ID is a valid Sentinel-2 L2A identifier."
            ) from exc
        raise

    if count == 0:
        raise ValueError(
            f"Scene {scene_id!r} not found in Earth Search. "
            "Verify the scene ID is a valid Sentinel-2 L2A identifier."
        )

    # Read back the parquet to extract metadata
    gdf = gpd.read_parquet(parquet_path)
    row = gdf.iloc[0]

    scene_dt = row.get("datetime")
    if scene_dt is None:
        scene_dt = datetime.now()
    elif hasattr(scene_dt, "to_pydatetime"):
        scene_dt = scene_dt.to_pydatetime()

    bbox = list(row.geometry.bounds)

    epsg = 4326
    proj_epsg = row.get("proj:epsg")
    if proj_epsg is not None:
        epsg = int(proj_epsg)

    cloud_cover = row.get("eo:cloud_cover")

    return SearchResult(
        scene_id=scene_id,
        parquet_path=parquet_path,
        datetime=scene_dt,
        bbox=bbox,
        epsg=epsg,
        cloud_cover=float(cloud_cover) if cloud_cover is not None else None,
        properties=dict(row.drop("geometry")),
    )


def load_scene(
    search_result: SearchResult,
    resolution: float = 10.0,
) -> np.ndarray:
    """Load a Sentinel-2 scene as a numpy array using lazycogs.

    Args:
        search_result: Result from search_scene().
        resolution: Target resolution in meters (default 10m).

    Returns:
        Numpy array of shape [C, H, W] with pixel values for all bands.

    Raises:
        ImportError: If lazycogs is not installed.
    """
    parquet_str = str(search_result.parquet_path)
    store = lazycogs.store_for(parquet_str, skip_signature=True)

    da = lazycogs.open(
        parquet_str,
        bbox=(
            search_result.bbox[0],
            search_result.bbox[1],
            search_result.bbox[2],
            search_result.bbox[3],
        ),
        crs=f"EPSG:{search_result.epsg}",
        resolution=resolution,
        bands=S2_BANDS,
        store=store,
    )

    # lazycogs returns (band, time, y, x) -- squeeze time for single scene
    data = da.compute()
    arr = data.values

    # Handle different dimension orderings
    if arr.ndim == 4:
        # (band, time, y, x) -> take first timestep -> (band, y, x)
        arr = arr[:, 0, :, :]
    elif arr.ndim == 3:
        # Already (band, y, x)
        pass
    else:
        raise ValueError(f"Unexpected array shape from lazycogs: {arr.shape}")

    # Ensure we have the right number of bands
    if arr.shape[0] != len(S2_BANDS):
        raise ValueError(
            f"Expected {len(S2_BANDS)} bands, got {arr.shape[0]}. Expected bands: {S2_BANDS}"
        )

    return arr.astype(np.float32)


def chip_array(
    pixels: np.ndarray,
    chip_size: int = 256,
    patch_size: int = 8,
) -> list[tuple[np.ndarray, int, int]]:
    """Chip a large array into model-ready patches.

    Chips the array into non-overlapping chips of chip_size x chip_size.
    Chips that don't fit evenly at the edges are dropped.

    Args:
        pixels: Array of shape [C, H, W].
        chip_size: Size of each chip in pixels (must be divisible by patch_size).
        patch_size: Model patch size (default 8).

    Returns:
        List of (chip, row_idx, col_idx) tuples where chip is [C, chip_size, chip_size].
    """
    if chip_size % patch_size != 0:
        raise ValueError(f"chip_size ({chip_size}) must be divisible by patch_size ({patch_size})")

    _C, H, W = pixels.shape
    n_rows = H // chip_size
    n_cols = W // chip_size

    if n_rows == 0 or n_cols == 0:
        raise ValueError(
            f"Image ({H}x{W}) is smaller than chip_size ({chip_size}). Use a smaller --chip-size."
        )

    chips = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * chip_size
            x0 = c * chip_size
            chip = pixels[:, y0 : y0 + chip_size, x0 : x0 + chip_size]
            chips.append((chip, r, c))

    return chips


def _compute_chip_bbox(
    scene_bbox: list[float],
    row: int,
    col: int,
    chip_size: int,
    resolution: float,
    full_height: int,
) -> tuple[float, float, float, float]:
    """Compute the bounding box for a chip in the scene's CRS.

    Args:
        scene_bbox: [minx, miny, maxx, maxy] of the full scene.
        row: Row index of the chip.
        col: Column index of the chip.
        chip_size: Chip size in pixels.
        resolution: Pixel resolution in CRS units (meters for UTM).
        full_height: Full height of the loaded array in pixels.

    Returns:
        (minx, miny, maxx, maxy) for this chip.
    """
    chip_extent = chip_size * resolution
    minx = scene_bbox[0] + col * chip_extent
    # y-axis is typically inverted in raster data (origin at top-left)
    maxy = scene_bbox[3] - row * chip_extent
    return (minx, maxy - chip_extent, minx + chip_extent, maxy)


def generate_embeddings(
    pixels: np.ndarray,
    encoder: Encoder,
    search_result: SearchResult,
    metadata: dict[str, PlatformMetadata] | None = None,
    chip_size: int = 256,
    batch_size: int = 16,
    device: str = "cpu",
    resolution: float = 10.0,
) -> list[EmbeddingChip]:
    """Generate Clay embeddings for all chips in an image.

    Chips the image, normalizes, encodes temporal/spatial information,
    and runs each batch through the encoder.

    Args:
        pixels: Raw pixel array of shape [C, H, W].
        encoder: Clay Encoder in eval mode.
        search_result: SearchResult with scene metadata.
        metadata: Sensor metadata (loaded automatically if None).
        chip_size: Chip size in pixels (default 256).
        batch_size: Number of chips per batch (default 16).
        device: Device for inference.
        resolution: Pixel resolution in meters.

    Returns:
        List of EmbeddingChip with embeddings and spatial metadata.
    """
    if metadata is None:
        metadata = load_metadata()

    sensor_meta = metadata[SENSOR]
    waves = torch.tensor(
        list(sensor_meta.bands.wavelength.values()),
        device=device,
    )
    gsd = torch.tensor(sensor_meta.gsd, dtype=torch.float32, device=device)

    # Encode temporal and spatial information
    time_enc = encode_timestamp(search_result.datetime).to(device)
    center_lat = (search_result.bbox[1] + search_result.bbox[3]) / 2
    center_lon = (search_result.bbox[0] + search_result.bbox[2]) / 2
    latlon_enc = encode_latlon(center_lat, center_lon).to(device)

    # Chip the array
    chips = chip_array(pixels, chip_size=chip_size)
    full_height = pixels.shape[1]

    results: list[EmbeddingChip] = []

    # Process in batches
    for batch_start in range(0, len(chips), batch_size):
        batch_chips = chips[batch_start : batch_start + batch_size]
        B = len(batch_chips)

        # Stack chips into a batch tensor
        batch_pixels = torch.from_numpy(np.stack([chip for chip, _, _ in batch_chips])).to(device)

        # Normalize
        batch_pixels = normalize(batch_pixels, SENSOR, metadata=metadata)

        # Build datacube
        datacube = {
            "pixels": batch_pixels,
            "time": time_enc.unsqueeze(0).expand(B, -1),
            "latlon": latlon_enc.unsqueeze(0).expand(B, -1),
            "gsd": gsd,
            "waves": waves,
        }

        # Run encoder
        with torch.no_grad():
            encoded, *_ = encoder(datacube)
            cls_embeddings = encoded[:, 0, :]  # [B, D]

        # Create EmbeddingChip for each result
        for i, (_, row, col) in enumerate(batch_chips):
            chip_bbox = _compute_chip_bbox(
                search_result.bbox, row, col, chip_size, resolution, full_height
            )
            results.append(
                EmbeddingChip(
                    embedding=cls_embeddings[i].cpu(),
                    row=row,
                    col=col,
                    bbox=chip_bbox,
                    epsg=search_result.epsg,
                    scene_id=search_result.scene_id,
                    datetime=search_result.datetime,
                )
            )

    return results


def save_embeddings_geoparquet(
    chips: list[EmbeddingChip],
    output_path: str | Path,
) -> Path:
    """Save embeddings to a GeoParquet file with spatial metadata.

    Each row contains the embedding vector, chip grid position,
    and a polygon geometry for the chip's spatial extent.

    Args:
        chips: List of EmbeddingChip from generate_embeddings().
        output_path: Path to write the GeoParquet file.

    Returns:
        Path to the written file.

    Raises:
        ImportError: If geopandas is not installed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for chip in chips:
        emb = chip.embedding.numpy()
        row_data = {
            "scene_id": chip.scene_id,
            "datetime": chip.datetime,
            "row": chip.row,
            "col": chip.col,
            "epsg": chip.epsg,
            "geometry": box(*chip.bbox),
        }
        # Store each embedding dimension as a separate column
        for dim_idx in range(emb.shape[0]):
            row_data[f"emb_{dim_idx:04d}"] = float(emb[dim_idx])
        rows.append(row_data)

    gdf = gpd.GeoDataFrame(rows, crs=f"EPSG:{chips[0].epsg}")
    gdf.to_parquet(output_path)

    return output_path


def run_pipeline(
    scene_id: str,
    ckpt_path: str,
    size: str = "large",
    device: str = "cpu",
    chip_size: int = 256,
    batch_size: int = 16,
    resolution: float = 10.0,
    output: str | None = None,
) -> Path:
    """Run the full Sentinel-2 inference pipeline.

    Searches for a scene, loads the COGs, generates embeddings for all
    chips, and saves to GeoParquet.

    Args:
        scene_id: Sentinel-2 scene ID from Earth Search.
        ckpt_path: Path to Clay checkpoint file.
        size: Model size (tiny, small, base, large).
        device: Device for inference (cpu, cuda).
        chip_size: Chip size in pixels.
        batch_size: Chips per batch.
        resolution: Target resolution in meters.
        output: Output path for GeoParquet. Defaults to <scene_id>_embeddings.parquet.

    Returns:
        Path to the output GeoParquet file.
    """
    if output is None:
        output = f"{scene_id}_embeddings.parquet"

    # Step 1: Search for the scene
    search_result = search_scene(scene_id)

    # Step 2: Load the scene bands
    pixels = load_scene(search_result, resolution=resolution)

    # Step 3: Load the model
    encoder = load_model(size=size, ckpt_path=ckpt_path, device=device)

    # Step 4: Generate embeddings
    chips = generate_embeddings(
        pixels=pixels,
        encoder=encoder,
        search_result=search_result,
        chip_size=chip_size,
        batch_size=batch_size,
        device=device,
        resolution=resolution,
    )

    # Step 5: Save to GeoParquet
    return save_embeddings_geoparquet(chips, output)
