import random
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
from tifffile import imread

EXPECTED_NR_OF_FILES_PER_TILE = 24
MONTHS = [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
]


def list_uniqe_ids(src: Path) -> list[str]:
    ids = list(set(dat.name.split("_")[0] for dat in src.glob("*.tif")))
    print(f"Found {len(ids)} unique tile ids")
    return ids


def process_data_for_id(
    id: str, feature_path: Path, cubes_path: Path, overwrite: bool
) -> None:
    if not overwrite and (cubes_path / f"biomasters_cube_{id}.npz").exists():
        print(f"Found existing file for {id}, skipping.")
        return
    data = []
    for month in MONTHS:
        data_month = []
        for platform in ["S1", "S2"]:
            feature_name = f"{id}_{platform}_{month}.tif"
            if not Path(feature_path / feature_name).exists():
                continue
            file_data = (
                imread(feature_path / feature_name).swapaxes(1, 2).swapaxes(0, 1)
            )
            ND1 = 0
            ND2 = -9999
            if platform == "S1":
                # Limit to first orbit (the other is mostly nodata)
                file_data = file_data[:2]
                file_data = np.ma.array(
                    file_data, mask=np.logical_or(file_data == ND1, file_data == ND2)
                )
            else:
                file_data = file_data[:10]
                file_data = np.ma.array(file_data, mask=file_data == ND1)
            data_month.append(file_data)

        data_month = np.ma.vstack(data_month)
        NR_OF_BANDS_EXPECTED = 12
        if data_month.shape[0] != NR_OF_BANDS_EXPECTED:
            continue
        data.append(data_month)

    cube = np.ma.array(data)
    mean_cube = np.ma.mean(cube, axis=0)

    if np.sum(mean_cube.mask):
        print("Nodata", np.sum(mean_cube.mask))
    NODATA_THRESHOLD = 1e5
    if np.sum(mean_cube.mask) > NODATA_THRESHOLD:
        print("Skipping due to lots of nodata")
        return

    np.savez_compressed(cubes_path / f"biomasters_cube_{id}.npz", cube=mean_cube)


@click.command()
@click.option(
    "--features",
    help="Folder with features (training or test)",
    type=click.Path(path_type=Path),
)
@click.option(
    "--cubes", help="Folder to write the datacubes", type=click.Path(path_type=Path)
)
@click.option(
    "--processes",
    default=1,
    help="How many processes to use for parallel processing",
    type=click.INT,
)
@click.option(
    "--sample",
    default=0.05,
    help="Fraction of original data to sample",
    type=click.FloatRange(0, 1),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing cubes",
)
def process(features, cubes, processes, sample, overwrite):
    """
    Combine tiff files into npz datacubes.

    The datacubes will contain the S1 vv/vh bands for asc and desc orbits,
    stacked with the first 10 S2 bands.
    """
    ids = list_uniqe_ids(features)

    if sample < 1:
        sample_length = int(len(ids) * sample)
        random.seed(42)
        ids = random.sample(ids, sample_length)

    print(f"Subsampled {len(ids)} tiles")

    if processes > 1:
        features = [features] * len(ids)
        cubes = [cubes] * len(ids)
        overwrite = [overwrite] * len(ids)
        with Pool(processes) as pl:
            pl.starmap(process_data_for_id, zip(ids, features, cubes, overwrite))
    else:
        for id in ids:
            process_data_for_id(id, features, cubes, overwrite)


if __name__ == "__main__":
    process()
