#!/usr/bin/env python3
# From https://github.com/alexgleith/sinergise-element84-sentinel-2-qa/blob/main/0-parse-inventory-element84.py
import csv
import gzip
import json
import sys

import boto3

SPECIAL_YEAR = "2019"
CUTOFF_YEAR = 2023

s3 = boto3.resource("s3")

bucket = "sentinel-cogs-inventory"
manifest_key = "sentinel-cogs/sentinel-cogs/2024-10-03T01-00Z/manifest.json"

print("Starting up...")


def log(comment):
    sys.stdout.write(f"\r{comment}")


# Stolen from https://alukach.com/posts/parsing-s3-inventory-output
def list_keys(bucket, manifest_key):
    manifest = json.load(s3.Object(bucket, manifest_key).get()["Body"])
    for item in manifest["files"]:
        gzip_obj = s3.Object(bucket_name=bucket, key=item["key"])
        buffer = gzip.open(gzip_obj.get()["Body"], mode="rt")
        reader = csv.reader(buffer)
        yield from reader


limit = 2
count = 0
valid = 0
log_every = 10000
cutoff_year = 2023

if __name__ == "__main__":
    # Parse zip file for all scenes
    with gzip.open("data/element84-tiles.list.gz", "wt") as text_file:
        for tiles_bucket, key, *rest in list_keys(bucket, manifest_key):
            if ".json" in key:
                c = key.split("/")
                # Counting scenes
                count += 1
                if count % log_every == 0:
                    log(f"Found {count} scenes...")
                tile = f"{c[1]}{c[2]}{c[3]}"
                text_file.write(f"s3://{tiles_bucket}/{key}\n")

    print(f"Found {count} scenes")

    # Reduce to 2023 and 20204
    with gzip.open("data/element84-tiles-2023.gz", "wt") as dst:
        with gzip.open("data/element84-tiles.list.gz") as fl:
            line = fl.readline()
            while line:
                line = line.decode().rstrip()
                c = line.split("/")
                # Skip data befor 2023. Some scenes from 2019 have the year
                # in a different part of the prefix.
                if c[4] == SPECIAL_YEAR:
                    line = fl.readline()
                    continue
                elif int(c[7]) < CUTOFF_YEAR:
                    line = fl.readline()
                    continue
                elif not line.endswith("L2A.json"):
                    line = fl.readline()
                    continue

                count += 1
                if count % log_every == 0:
                    log(f"Found {count} scenes... {line}")

                dst.write(line + "\n")
                line = fl.readline()
