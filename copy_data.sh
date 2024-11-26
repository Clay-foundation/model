#!/bin/bash

# Define source and destination directories
src="/fsx"
dest="data/pretrain"

# Create the destination directory if it doesn't exist
mkdir -p "$dest"

# Find all directories in the source directory
find "$src" -type d -print0 | while IFS= read -r -d '' dir; do
    # Create corresponding directory in the destination
    newdir="$dest${dir#$src}"
    mkdir -p "$newdir"

    # Copy the first 100 files from the source directory to the new directory
    find "$dir" -maxdepth 1 -type f -print0 | head -z -n 100 | xargs -0 -I{} cp {} "$newdir"
done
