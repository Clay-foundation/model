# How to run clay over custom AOIs

This section shows in a few simple steps how the clay model can be run for
custom AOIs and over custom date ranges.

## Prepare folder strucutre for data

```bash
# Move into the model repository
cd /path/to/repository/model/

# Ensure data sub-directories exist
mdkir data/mgrs
mdkir data/chips
mdkir data/embeddings
```

## Download global list of MGRS tiles

The first step is to download a list of globally available MGRS tiles. A full
list of MGRS tiles has been created as part of the landcover based sampling
strategy. The file is [sourced from a complete MGRS tile list](https://github.com/Clay-foundation/model/blob/main/scripts/landcover.sh#L7),
and then [itersected with the WorldCover landcover](https://github.com/Clay-foundation/model/blob/main/scripts/landcover.py)
layer, outputting the `mgrs_full.fgb` file that is used below.

```bash
wget https://clay-mgrs-samples.s3.amazonaws.com/mgrs_full.fgb -O data/mgrs/mgrs_full.fgb
```

## Create a Geopandas dataframe with MGRS tiles over the AOI

This example uses a bounding box over the area around Puri, India, to
filter the global list of MGRS tiles. The intersected MGRS tiles are
then stored into a new dataset with the reduced list. The reduced list
 will be used by the `datacube.py` script for creating imagery chips.

```python
import geopandas as gpd
import pandas as pd
from shapely import box


mgrs = gpd.read_file("data/mgrs/mgrs_full.fgb")
print(f"Loaded {len(mgrs)} MGRS grid cells.")

aoi = gpd.GeoDataFrame(
    pd.DataFrame(["Puri"], columns=["Region"]),
    crs="EPSG:4326",
    geometry=[box(85.0503, 19.4949, 86.1042, 20.5642)],
)
mgrs_aoi = mgrs.overlay(aoi)

# Rename the name column to use lowercase letters for the datacube script to
# pick upthe MGRS tile name.
mgrs_aoi = mgrs_aoi.rename(columns={"Name": "name"})

print(f"Found {len(mgrs_aoi)} matching MGRS tiles over the AOI.")

mgrs_aoi.to_file("data/mgrs/mgrs_aoi.fgb")
```

## Use the datacube.py script to download imagery

This will select the MGRS tiles that intersect with your AOI. The processing
will then happen for each of the MGRS tiles. This will most likely provide
slightly more data than the AOI itself, as the whole tile data will downloaded
for each matched MGRS tile.

Each run of th datacube script will take an index as input, which is the index
of the MGRS tile within the input file. This is why we need to download the
data in a loop.

A list of date ranges can be specified. The script will look for the least
cloudy Sentinel-2 scene for each date range, and match Sentinel-1 dates near
the identified Sentinel-2 dates.

The output folder can be specified as a local folder, or a bucket can be
specified to upload the data to S3.

Note that for the script to run, a Microsoft Planetary Computer token needs
to be set up, consult the [Planetary Computer SDK](https://github.com/microsoft/planetary-computer-sdk-for-python)
documentation on how to set up the token.

By default, the datacube script will download all the data available for each
MGRS tile it processes. So the output might include imagery chips that are
outside of the AOI specified.

To speed up processing in the example below, we use the subset argument to
reduce each MGRS tile to a small pixel window. When subsetting, the script
will only download a fraction of each MGRS tile. This will lead to discontinous
datasets and should not be used in a real use case. Remove the subset argument
when using the script for a real world application, where all the data should
be downloaded for each MGRS tile.

```bash
for i in {0..5}; do

python scripts/datacube.py \
    --sample data/mgrs/mgrs_aoi.fgb \
    --localpath data/chips  \
    --index $i \
    --dateranges 2020-01-01/2020-04-01,2021-06-01/2021-09-15 \
    --subset 1500,1500,2524,2524;

done
```

## Create the embeddings for each training chip

The checkpoints can be accessed directly from Hugging Face
at https://huggingface.co/made-with-clay/Clay.

The following command will run the model to create the embeddings,
and automatically download and cache the model weights.

```bash
wandb disabled
python trainer.py predict \
    --ckpt_path=https://huggingface.co/made-with-clay/Clay/resolve/main/Clay_v0.1_epoch-24_val-loss-0.46.ckpt \
    --trainer.precision=16-mixed \
    --data.data_dir=/home/tam/Desktop/aoitiles \
    --data.batch_size=2 \
    --data.num_workers=8
```
