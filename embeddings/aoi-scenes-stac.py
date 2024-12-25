import gzip
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

geom_geojson = "data/USA.geojson"
year_start = 2022
year_end = 2024
output_csv = f"data/stac_manifest_{geom_geojson.split('/')[-1].split('.')[0]}_"+\
    f"sentinel_{year_start}_{year_end}-2.csv.gz"
print(output_csv)

HTTP_OK = 200

def create_session():
    """Create a requests session with retry logic and connection pooling"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
    return session

def process_geometry(session, stac_api, aoi_geom, i, total, output_file):
    """Process a single geometry and write scenes directly to file"""
    scenes_count = 0
    page = 1
    params = {
        "collections": ["sentinel-2-l2a"],
        "intersects": aoi_geom,
        "datetime": f"{year_start}-01-01T00:00:00Z/{year_end}-12-31T23:59:59Z",
        "limit": 100,
    }
    
    print(f"\nProcessing geometry {i+1}/{total}")
    
    while True:
        print(f"  Fetching page {page} for geometry {i+1}/{total}...", end="\r")
        response = session.post(stac_api, json=params)
        
        if response.status_code != HTTP_OK:
            print(f"\n  Error processing geometry {i+1}/{total}: HTTP {response.status_code}")
            print(f"  Response: {response.text[:200]}...")
            break
            
        result = response.json()
        new_scenes = [
            link["href"] for feature in result.get("features", [])
            for link in feature["links"] if link["rel"] == "canonical"
        ]
        
        # Write scenes directly to file
        if new_scenes:
            output_file.write("\n".join(new_scenes) + "\n")
            scenes_count += len(new_scenes)
            print(f"  Page {page}: Wrote {len(new_scenes)} scenes for geometry {i+1}/{total}")
        
        next_link = next((link for link in result.get("links", []) if link["rel"] == "next"), None)
        if not next_link:
            print(f"  Completed geometry {i+1}/{total}: Found {scenes_count} total scenes")
            break
            
        params = next_link["body"]
        page += 1
    
    return scenes_count

def get_sentinel_scenes_brazil():
    print(f"\nStarting search for Sentinel scenes from {year_start} to {year_end}")
    stac_api = "https://earth-search.aws.element84.com/v1/search"

    print(f"Loading geometries from {geom_geojson}")
    with open(geom_geojson) as f:
        data = json.load(f)
        if data.get("type") == "FeatureCollection":
            aoi_geoms = [feature["geometry"] for feature in data["features"]]
            print(f"Loaded {len(aoi_geoms)} geometries from FeatureCollection")
        else:
            aoi_geoms = [data["geometry"]]
            print("Loaded single geometry from Feature")

    print("\nInitializing HTTP session with retry logic")
    session = create_session()
    total_scenes = 0
    
    print(f"\nProcessing geometries with {min(5, len(aoi_geoms))} concurrent workers")
    print(f"Writing scenes to: {output_csv}")
    
    with gzip.open(output_csv, "wt", newline="") as f:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_geom = {
                executor.submit(process_geometry, session, stac_api, geom, i, len(aoi_geoms), f): i 
                for i, geom in enumerate(aoi_geoms)
            }
            
            for future in as_completed(future_to_geom):
                try:
                    scenes_count = future.result()
                    total_scenes += scenes_count
                    print(f"\nRunning total: {total_scenes} scenes written")
                except Exception as e:
                    print(f"\nError processing geometry: {str(e)}")
    
    return total_scenes

# Main execution
print(f"Output will be saved to: {output_csv}")
total_scenes = get_sentinel_scenes_brazil()
print(f"\nComplete! Total scenes written: {total_scenes}")
print(f"Scenes saved to: {output_csv}")
