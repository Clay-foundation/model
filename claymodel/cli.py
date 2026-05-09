"""Clay model CLI."""

import time as time_mod

import click
import torch

from claymodel.api import load_metadata
from claymodel.model import Encoder


@click.group()
@click.version_option(package_name="claymodel")
def cli() -> None:
    """Clay Foundation Model CLI."""


@cli.command()
@click.option("--sensor", default=None, help="Show details for a specific sensor")
def info(sensor: str | None) -> None:
    """Show model and sensor information."""
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
        if s.rgb_indices is None:
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
def benchmark(device: str, size: int) -> None:
    """Run a smoke test and print timing."""
    click.echo(f"Running benchmark (device={device}, chip_size={size})")

    metadata = load_metadata()
    sensor = "sentinel-2-l2a"
    n_bands = len(metadata[sensor].band_order)

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

    waves = torch.tensor(list(metadata[sensor].bands.wavelength.values()), device=device)

    datacube = {
        "pixels": torch.randn(1, n_bands, size, size, device=device),
        "time": torch.zeros(1, 4, device=device),
        "latlon": torch.zeros(1, 4, device=device),
        "gsd": torch.tensor(10.0, device=device),
        "waves": waves,
    }

    with torch.no_grad():
        encoder(datacube)

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
