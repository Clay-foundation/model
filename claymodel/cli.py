"""Clay model CLI."""

import importlib.util

import click

from claymodel.api import load_metadata, load_model


def _check_inference_deps() -> None:
    """Check that inference optional dependencies are installed."""
    missing = [
        pkg
        for pkg in ("rustac", "lazycogs", "geopandas", "shapely")
        if importlib.util.find_spec(pkg) is None
    ]
    if missing:
        raise click.ClickException(
            f"Missing inference dependencies: {', '.join(missing)}\n"
            "Install them with: pip install claymodel[inference]"
        )


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
@click.argument("scene_id")
@click.option("--ckpt", required=True, help="Path to Clay checkpoint file")
@click.option(
    "--size",
    default="large",
    type=click.Choice(["tiny", "small", "base", "large"]),
    help="Model size (default: large)",
)
@click.option("--device", default="cpu", help="Device for inference (cpu, cuda)")
@click.option("--chip-size", default=256, type=int, help="Chip size in pixels (default: 256)")
@click.option("--batch-size", default=16, type=int, help="Chips per batch (default: 16)")
@click.option("--resolution", default=10.0, type=float, help="Target resolution in meters")
@click.option("--output", default=None, help="Output GeoParquet path")
def embed(
    scene_id: str,
    ckpt: str,
    size: str,
    device: str,
    chip_size: int,
    batch_size: int,
    resolution: float,
    output: str | None,
) -> None:
    """Generate Clay embeddings for a Sentinel-2 scene.

    SCENE_ID is a Sentinel-2 scene identifier from Earth Search STAC
    (e.g. S2A_MSIL2A_20240101T100321_N0510_R122_T33UUP_20240101T120401).

    Searches Earth Search, loads COG bands via lazycogs, chips into patches,
    runs the Clay encoder, and saves embeddings as GeoParquet.

    Requires: pip install claymodel[inference]
    """
    _check_inference_deps()

    from claymodel.inference.pipeline import (
        generate_embeddings,
        load_scene,
        save_embeddings_geoparquet,
        search_scene,
    )

    click.echo(f"Scene: {scene_id}")
    click.echo(f"Model: {size} (checkpoint: {ckpt})")
    click.echo(f"Device: {device}")
    click.echo(f"Chip size: {chip_size}px at {resolution}m resolution")
    click.echo()

    click.echo("Searching Earth Search STAC...")
    try:
        search_result = search_scene(scene_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"  Found scene: {search_result.scene_id}")
    click.echo(f"  Date: {search_result.datetime}")
    click.echo(f"  EPSG: {search_result.epsg}")
    if search_result.cloud_cover is not None:
        click.echo(f"  Cloud cover: {search_result.cloud_cover:.1f}%")
    click.echo()

    click.echo("Loading scene bands via lazycogs...")
    pixels = load_scene(search_result, resolution=resolution)

    _C, H, W = pixels.shape
    n_chips = (H // chip_size) * (W // chip_size)
    click.echo(f"  Image size: {_C} bands x {H} x {W} pixels")
    click.echo(f"  Chips: {n_chips} ({H // chip_size} rows x {W // chip_size} cols)")
    click.echo()

    click.echo("Loading Clay encoder...")
    encoder = load_model(size=size, ckpt_path=ckpt, device=device)
    click.echo(f"  Encoder dim: {encoder.dim}")
    click.echo()

    click.echo("Generating embeddings...")
    chips = generate_embeddings(
        pixels=pixels,
        encoder=encoder,
        search_result=search_result,
        chip_size=chip_size,
        batch_size=batch_size,
        device=device,
        resolution=resolution,
    )
    click.echo(f"  Generated {len(chips)} embeddings of dim {chips[0].embedding.shape[0]}")
    click.echo()

    click.echo("Saving to GeoParquet...")
    output_path = save_embeddings_geoparquet(
        chips,
        output or f"{scene_id}_embeddings.parquet",
    )
    click.echo(f"  Saved to: {output_path}")
    click.echo()
    click.echo("Done.")


if __name__ == "__main__":
    cli()
