"""
Chesapeake CVPR Data Processing Script
======================================

This script processes GeoTIFF files from the Chesapeake CVPR dataset to create image chips for segmentation tasks.

Dataset Source:
---------------
Chesapeake CVPR data from LILA: https://lila.science/datasets/chesapeakelandcover
For this experiment, we will use images from NY.

Notes:
------
1. Only copy *_lc.tif & *_naip-new.tif files that we will use for our segmentation downstream task.
   Using s5cmd for this: https://github.com/peak/s5cmd
   - Train: s5cmd cp --include "*_lc.tif" --include "*_naip-new.tif" "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-train_tiles/*" data/cvpr/files/train/
   - Val: s5cmd cp --include "*_lc.tif" --include "*_naip-new.tif" "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-val_tiles/*" data/cvpr/files/val/

2. We will create chips of size `224 x 224` to feed them to the model, feel free to experiment with other chip sizes as well.
   Run the script as follows:
   python script.py <data_dir> <output_dir> <chip_size>

   Example:
   python script.py data/cvpr/files data/cvpr/ny 224
"""

import os
import sys
from pathlib import Path

import numpy as np
import rasterio as rio


def read_and_chip(file_path, chip_size, output_dir):
    """
    Reads a GeoTIFF file, creates chips of specified size, and saves them as numpy arrays.

    Args:
        file_path (str or Path): Path to the GeoTIFF file.
        chip_size (int): Size of the square chips.
        output_dir (str or Path): Directory to save the chips.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rio.open(file_path) as src:
        data = src.read()

        n_chips_x = src.width // chip_size
        n_chips_y = src.height // chip_size

        chip_number = 0
        for i in range(n_chips_x):
            for j in range(n_chips_y):
                x1, y1 = i * chip_size, j * chip_size
                x2, y2 = x1 + chip_size, y1 + chip_size

                chip = data[:, y1:y2, x1:x2]
                chip_path = os.path.join(
                    output_dir, f"{Path(file_path).stem}_chip_{chip_number}.npy"
                )
                np.save(chip_path, chip)
                chip_number += 1


def process_files(file_paths, output_dir, chip_size):
    """
    Processes a list of files, creating chips and saving them.

    Args:
        file_paths (list of Path): List of paths to the GeoTIFF files.
        output_dir (str or Path): Directory to save the chips.
        chip_size (int): Size of the square chips.
    """
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        read_and_chip(file_path, chip_size, output_dir)


def main():
    """
    Main function to process files and create chips.
    Expects three command line arguments:
        - data_dir: Directory containing the input GeoTIFF files.
        - output_dir: Directory to save the output chips.
        - chip_size: Size of the square chips.
    """
    if len(sys.argv) != 4:
        print("Usage: python script.py <data_dir> <output_dir> <chip_size>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    chip_size = int(sys.argv[3])

    train_image_paths = list((data_dir / "train").glob("*_naip-new.tif"))
    val_image_paths = list((data_dir / "val").glob("*_naip-new.tif"))
    train_label_paths = list((data_dir / "train").glob("*_lc.tif"))
    val_label_paths = list((data_dir / "val").glob("*_lc.tif"))

    process_files(train_image_paths, output_dir / "train/chips", chip_size)
    process_files(val_image_paths, output_dir / "val/chips", chip_size)
    process_files(train_label_paths, output_dir / "train/labels", chip_size)
    process_files(val_label_paths, output_dir / "val/labels", chip_size)


if __name__ == "__main__":
    main()
