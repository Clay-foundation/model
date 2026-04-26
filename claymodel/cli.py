"""
Clay Foundation Model CLI.

Usage:
    clay embed image.tif --sensor sentinel-2-l2a --ckpt clay-v1.5.ckpt
    clay info
    clay info --sensor sentinel-1-rtc
    clay benchmark

Requires the [cli] extras: pip install claymodel[cli]
"""

import time as time_mod

import click
import torch

from claymodel.api import embed as embed_fn
from claymodel.api import load_metadata
from claymodel.model import Encoder


@click.group()
@click.version_option(package_name="claymodel")
def cli():
    """Clay Foundation Model CLI."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--sensor", required=True, help="Sensor name (e.g., sentinel-2-l2a)")
@click.option("--ckpt", required=True, help="Path to Clay checkpoint file")
@click.option(
    "--output", "-o", default=None, help="Output path (.parquet or .geoparquet)"
)
@click.option("--device", default="cpu", help="Device (cpu, cuda, etc.)")
@click.option(
    "--min-valid",
    default=0.0,
    type=float,
    help="Skip chips with valid_fraction below this threshold",
)
def embed(input_path, sensor, ckpt, output, device, min_valid):
    """Generate embeddings from a GeoTIFF file.

    Reads the input file, normalizes using sensor-specific statistics,
    runs the Clay encoder, and outputs embeddings.

    Example:
        clay embed image.tif --sensor sentinel-2-l2a --ckpt clay-v1.5.ckpt
    """
    click.echo(f"Embedding {input_path} (sensor={sensor}, device={device})")

    start = time_mod.perf_counter()
    result = embed_fn(input_path, sensor=sensor, ckpt_path=ckpt, device=device)
    elapsed = time_mod.perf_counter() - start

    click.echo(f"Embeddings shape: {result.shape} ({elapsed:.2f}s)")

    if output:
        result.to_geoparquet(output)
        click.echo(f"Saved to {output}")


@cli.command()
@click.option("--sensor", default=None, help="Show details for a specific sensor")
def info(sensor):
    """Show model and sensor information.

    Without --sensor, lists all supported sensors.
    With --sensor, shows band details, wavelengths, and normalization stats.

    Example:
        clay info
        clay info --sensor sentinel-2-l2a
    """
    metadata = load_metadata()

    if sensor:
        if sensor not in metadata:
            click.echo(f"Unknown sensor: {sensor}")
            click.echo(f"Available: {', '.join(metadata.keys())}")
            raise SystemExit(1)

        s = metadata[sensor]
        click.echo(f"\n{sensor}")
        click.echo(f"  GSD: {s.gsd}m")
        click.echo(f"  Bands: {len(s.band_order)}")
        click.echo(f"  Band order: {', '.join(s.band_order)}")
        click.echo("\n  Wavelengths (um):")
        for band, wl in s.bands.wavelength.items():
            click.echo(f"    {band}: {wl}")
        click.echo("\n  Normalization (mean / std):")
        for band in s.band_order:
            m = s.bands.mean[band]
            sd = s.bands.std[band]
            click.echo(f"    {band}: {m} / {sd}")
        if not hasattr(s, "rgb_indices"):
            click.echo("\n  Note: No RGB indices (SAR sensor)")
    else:
        click.echo("\nClay v1.5 Supported Sensors:")
        click.echo("-" * 50)
        for name in metadata:
            s = metadata[name]
            n_bands = len(s.band_order)
            click.echo(f"  {name:25s} {n_bands:2d} bands  {s.gsd:6.1f}m GSD")
        click.echo(f"\nTotal: {len(metadata)} sensors")
        click.echo("\nUse 'clay info --sensor <name>' for details.")


@cli.command()
@click.option("--device", default="cpu", help="Device (cpu, cuda)")
@click.option("--size", default=64, type=int, help="Chip size for benchmark")
def benchmark(device, size):
    """Run a smoke test and print timing.

    Creates a tiny model with random weights, runs a forward pass,
    and reports timing and PASS/FAIL status. No checkpoint needed.

    Example:
        clay benchmark
        clay benchmark --device cuda --size 256
    """
    click.echo(f"Running benchmark (device={device}, chip_size={size})")

    metadata = load_metadata()
    sensor = "sentinel-2-l2a"
    n_bands = len(metadata[sensor].band_order)

    # Create tiny encoder (random weights, no checkpoint)
    encoder = Encoder(
        mask_ratio=0.0,
        patch_size=8,
        shuffle=False,
        dim=192,
        depth=2,
        heads=4,
        dim_head=48,
        mlp_ratio=2,
    ).to(device)
    encoder.eval()

    waves = torch.tensor(
        list(metadata[sensor].bands.wavelength.values()), device=device
    )

    datacube = {
        "pixels": torch.randn(1, n_bands, size, size, device=device),
        "time": torch.zeros(1, 4, device=device),
        "latlon": torch.zeros(1, 4, device=device),
        "gsd": torch.tensor(10.0, device=device),
        "waves": waves,
    }

    # Warmup
    with torch.no_grad():
        encoder(datacube)

    # Benchmark
    n_runs = 10
    start = time_mod.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            encoded, *_ = encoder(datacube)
    elapsed = time_mod.perf_counter() - start

    avg_ms = (elapsed / n_runs) * 1000
    cls_embedding = encoded[:, 0, :]

    click.echo("\nResults:")
    click.echo(f"  Encoder dim: {encoder.dim}")
    click.echo(f"  Input: [{1}, {n_bands}, {size}, {size}]")
    click.echo(f"  Output CLS: {list(cls_embedding.shape)}")
    click.echo(f"  Avg time: {avg_ms:.1f}ms ({n_runs} runs)")
    click.echo(f"  Device: {device}")

    # Sanity checks
    passed = True
    if cls_embedding.shape != (1, 192):
        click.echo(f"  FAIL: Expected shape [1, 192], got {list(cls_embedding.shape)}")
        passed = False
    if torch.isnan(cls_embedding).any():
        click.echo("  FAIL: NaN in output")
        passed = False
    if torch.isinf(cls_embedding).any():
        click.echo("  FAIL: Inf in output")
        passed = False

    if passed:
        click.echo("\n  PASS")
    else:
        click.echo("\n  FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
