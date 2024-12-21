import gzip
import json

import requests

# HTTP status codes
HTTP_OK = 200


def get_sentinel_scenes_brazil():
    # Earth Search STAC API endpoint
    stac_api = "https://earth-search.aws.element84.com/v1/search"

    # Load Brazil geometry from GeoJSON
    with open("data/brazil.geojson") as f:
        brazil_geom = json.load(f)["geometry"]

    # Search parameters
    params = {
        "collections": ["sentinel-2-l2a"],
        "intersects": brazil_geom,
        "datetime": "2023-01-01T00:00:00Z/2024-12-31T23:59:59Z",
        "limit": 100,  # Adjust based on needs
    }

    scenes = []

    while True:
        response = requests.post(stac_api, json=params)
        if response.status_code != HTTP_OK:
            print(f"Error: {response.status_code}")
            break

        result = response.json()

        # Process features
        for feature in result.get("features", []):
            # Find canonical link
            canonical_link = next(
                (
                    link["href"]
                    for link in feature["links"]
                    if link["rel"] == "canonical"
                ),
                None,
            )
            if canonical_link:
                scenes.append(canonical_link)

        # Handle pagination
        next_link = next(
            (link for link in result.get("links", []) if link["rel"] == "next"), None
        )
        if not next_link:
            break

        params = next_link["body"]
        print(len(scenes), scenes[-1], end="\r")
    return scenes


# Save scenes to CSV and print total
scenes = get_sentinel_scenes_brazil()
csv_path = "data/stac_manifest_brazil_sentinel_2023_2024.csv.gz"

with gzip.open(csv_path, "wt", newline="") as f:
    f.write("\n".join(scenes))

print(f"\nTotal scenes found: {len(scenes)}")
print(f"Scenes saved to: {csv_path}")
