import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def check_and_delete_npz(file_path):
    try:
        # Attempt to load the .npz file using numpy
        data = np.load(file_path)

        # Check if the 'pixel' key exists and has shape 128 in the 0th dimension
        if "pixels" in data:
            if data["pixels"].shape[0] != 128:  # noqa: PLR2004
                os.remove(file_path)
                return (
                    None,
                    f"Invalid shape (not 128 in 0th dim): {file_path} - Deleted",
                )
            else:
                return f"Valid: {file_path}", None
        else:
            os.remove(file_path)
            return None, f"'pixels' key missing: {file_path} - Deleted"

    except Exception as e:
        os.remove(file_path)
        return None, f"Invalid (Exception): {file_path} - {str(e)} - Deleted"


def process_directory_in_parallel(directory, max_workers=4):
    invalid_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".npz"):
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(check_and_delete_npz, file_path))

        for future in as_completed(futures):
            valid_msg, invalid_msg = future.result()
            if valid_msg:
                print(valid_msg)
            if invalid_msg:
                print(invalid_msg)
                invalid_files.append(invalid_msg)

    return invalid_files


# Replace 'your_directory_path' with the path to the directory you want to check
invalid_files = process_directory_in_parallel("/fsx", max_workers=24)

if invalid_files:
    print("\nInvalid or corrupted .npz files found and deleted:")
    for file in invalid_files:
        print(file)
else:
    print("\nAll .npz files are valid and meet the shape criteria for 'pixel' key.")
