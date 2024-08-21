import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def split_npz_file(file_path):
    # Load the .npz file
    with np.load(file_path) as data:
        # Check if the file has the required batch size of 128
        if "pixels" in data and data["pixels"].shape[0] == 128:  # noqa: PLR2004
            # Extract all arrays
            keys = data.files
            arrays = {key: data[key] for key in keys}

            # Determine the batch size and the number of splits
            batch_size = 32
            num_splits = 4  # Since we want to split into 4 files, each with 32 samples

            # Split and save the smaller .npz files
            for i in range(num_splits):
                split_data = {
                    key: value[i * batch_size : (i + 1) * batch_size]
                    for key, value in arrays.items()
                }
                split_file_path = file_path.replace(".npz", f"_{i}.npz")
                np.savez(split_file_path, **split_data)
                print(f"Saved {split_file_path}")

            # Delete the original file
            os.remove(file_path)
            print(f"Deleted original file: {file_path}")
        else:
            print(f"Skipped {file_path}: Does not have a batch size of 128")


def process_directory(root_dir):
    # Collect all .npz files
    npz_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".npz"):
                file_path = os.path.join(dirpath, filename)
                npz_files.append(file_path)

    # Process files in parallel
    with ProcessPoolExecutor() as executor:
        executor.map(split_npz_file, npz_files)


# Example usage
root_dir = "/fsx"
process_directory(root_dir)
