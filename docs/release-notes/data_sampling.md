(training-data)=
# Training Data

This section describes how we created the training dataset for the clay model.

## Data sources selection

The goal for the Clay model is for it to be as general as possible. It should be able to accept data from any platform coming from satellites, aerial, or drone platforms. For this to be possible, clearly the model design is the basis. Drawing inspiration from earlier works on Foundation models like Prithvi, SatMAE, ScaleMAE, DOFA, and SpectralGPT, we have developed a model architecture capable of accepting inputs of diverse spectral bands and resolutions in different sizes.

To train such a model, it is necessary to create a training dataset
that contains data from multiple platforms, and is as varied as possible in terms of

- spectral band definitions
- spatial distribution
- temporal distribution
- ground sampling distance

To achieve this we have first complied a [list of possible input platforms](https://github.com/Clay-foundation/model/issues/128). The list of candidate systems is rather long, and will be growing in the future. To reduce complexity, we have converged to a shorter list of platforms for the first round of model training.

Criteria was availability in the cloud, existence of STAC catalogs, and cloud optimized formats. This resulted in the following list of systems that we have
included in the training for Clay v1


| Platform | Spatial Coverage | Spectral bands | GSD (meters) |
---------|------------------|----------------|--------------|
| Landsat 8 and 9 | Global | 6 optical bands | 30 |
| Sentinel 2 L2A | Global | 10 optical bands | 10 |
| Sentinel 1 RTC | Global | 2 radar bands | 10 |
| NAIP | USA | 4 optical bands | < 1 |
| LINZ | New Zealand | 3 optical bands | < 0.5
| MODIS | Global | 7 bands | 500 |

## Sampling strategy

Once imagery sources are selected, the next step is to develop a sampling strategy. We are not able to process the entire archive, and so it is important to select the right subset of the archives for training.

Our driving principle is that the model should learn natural features as well as human made features. Human made features are smaller and less evenly distributed in many cases. This has driven some of the decisions for the sampling, as described below.

### Global sampling

We created a single sampling strategy for all four global satellite systems that we included in the model training (Sentinel 1 and 2, and Landsat 8 and 9). To create a balanced dataset for model training, we used a sampling strategy
based on land cover classes from the [ESA WorldCover](https://esa-worldcover.org/)
layer.

Our unit of analysis for sampling was the MGRS tile, the global tiling scheme
that is used for distributing Sentinel-2 imagery. For each MGRS tile, we
computed landcover statistics for all the classes in the WorldCover layer. To
speed up processing, we used the third level overview in the WorldCover layer,
which has a spatial resolution of 80 meters.

The goal of the landcover sampling was to ensure coverage of each class at
a reasonable level. For each class, we selected a number of random MGRS tiles
out of the subset of MGRS tiles with the highest fraction of that class present.

As an example, for "Wetlands" we selected 50 random ones out of the MGRS tiles
with the highest wetland fraction globally. For the Built-up class on the other
hand we selected the 400 most urban MGRS tiles.

In addition to the landcover classes, we also added diversity by selecting 500
tiles out of the 3000 tiles with the highest count of land cover classes present
in the tile.

After selecting MGRS tiles for each of these criteria, we removed duplicates.

The following table summarizes the selection criteria for each class.

| Class | Nr of Tiles | From highest |
|---|---|---|
Diversity | 400 | 2000
Built-up | 300 | 300
Built-up | 1000 | 1500
Herbaceous wetland | 50 | 500
Mangroves | 50 | 500
Moss and lichen | 50 | 500
Cropland | 800 | 3600
Tree cover | 150 | 750
Shrubland | 100 | 500
Grassland | 200 | 500
Bare / sparse vegetation | 50 | 500
Snow and Ice | 25 | 500
Permanent water bodies | 50 | 1000

This resulted in a sample of 2728 MGRS tiles total in our sample. The resulting sample file can be downloaded from the following link

https://clay-mgrs-samples.s3.amazonaws.com/mgrs_sample_v02.fgb

We used these locations for all of the global platforms. For more details about how exactly we implemented the sample selection, review the corresponding [stacchip processors](https://github.com/Clay-foundation/stacchip/blob/main/stacchip/processors/).

### Landsat 8 and 9 sampling strategy

To further increase variety in the dataset, we used both L1 and L2 products for training. For each location and each level of the platform, we selected one random year between 2018 and 2023, and used the least cloudy scenes in each quarter of the selected year.

### Sentinel-2 sampling strategy

For each location we selected two random years between 2018 and 2023, and for each year we used the least cloudy scene in each quarter.


### NAIP sampling strategy

The sampling strategy for [NAIP](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip) was based on [Natural Earth](https://www.naturalearthdata.com) data. The sample includes all popluated places, protected
areas and parks, airports, and ports. In addition, we sampled one random point
along each river, and one random location within each lake that is registered
in Natural Earth. Finally, we sampled 4000 random points. All data was
filtered to be within the CONUS region.

### LINZ sampling strategy

For [LINZ](https://github.com/linz/imagery) we used simple random subsampling because there is no STAC api to do spatial search with. We selected a random subset of all scenes for the different sub-collections that are available for LINZ.

More specifically, we randomly select 50% the scenes, with a minimum of 10
and a maximum of 2000 scenes for each catalog that was included.
We selected the latest imagery for each of the available regions
of new zealand. The list of catalogs is in the linz processor file.

### MODIS sampling strategy

For MODIS we used the [Surface Reflectance 8-Day (500m)](https://planetarycomputer.microsoft.com/dataset/modis-09A1-061)
product. The data is distributed in SIN grid tiles. We included all SIN grid
tiles that do not have any nodata inside. The selected SIN grid tiles are then
transform to EPSG:3857 for all tiles. This results in some variation between the
nominal resolution, although the original resolution from the SIN projection is
500 meters. For input to the model, we assumed the 500m resolution as a fixed
resolution size for all tiles.

Algorithm to determine which tiles do not have nodata is shown in the code block
below. This resulted in 233 SIN grid tiles to be selected. For each of these
we sampled the first STAC search result for each month in each year from 2018
until 2023. This therefore resulted in 72 (`6 years * 12 months`) separate scenes
for each of the 233 SIN grid tiles.

Script for selection of SIN grid tiles included in the sampling:

```python
from multiprocessing import Pool
import rasterio
import planetary_computer as pc
import pystac_client
import numpy as np

SIN_GRID_TILES = []
for i in SIN_VERTICAL_RANGE:
    for j in SIN_HORIZONTAL_RANGE:
        SIN_GRID_TILES.append((i, j))

def evaluate_nodata(i, j):
    catalog = pystac_client.Client.open(STAC_API, modifier=pc.sign_inplace)
    items = catalog.search(
        collections=[COLLECTION],
        query={
            "modis:vertical-tile": {
                "eq": i,
            },
            "modis:horizontal-tile": {
                "eq": j,
            },
        },
        max_items=1,
    )
    item = list(items.item_collection())[0]

    with rasterio.open(item.assets["sur_refl_b01"].href) as src:
        data = src.read()

    nodata = np.sum(data == -28672)

    if nodata == 0:
        print(i, j)
        return i, j

if __name__ == '__main__':
    with Pool(16) as p:
        indexes = p.starmap(evaluate_nodata, SIN_GRID_TILES)
    print("done")
    print(indexes)
```

## Data preparation

To be able to include multiple platforms in model training, we worked on a standardisation of the processing pipeline. The goal for this was to develop a framework that can be used to collect data from a large variety of formats and locations in a consistent way. For this we developed [stacchip](https://clay-foundation.github.io/stacchip/), a library to help preparing training data images. Please consult the documentation of the library to know more, but at a high level the goals of stacchip are

- Keeping the data in original format for as long as possible
- Scalable extendable indexing of chips
- Indexing processors for different platforms
- Chipping utility that takes the index and dynamically creates images for training
- Use geoparquet: fast storage option and easy to combine indexes from platforms
- Can be used for training and inference on the fly

## Dataset size

Using stacchip, we created a dataset with a size of 33.8 TB of imagery, with about 70 million chips created. The following table shows the distribution of imagery chips used for Clay v1 training.

| Source | Number of chips |
| ------ | --------------- |
<<<<<<< HEAD
| NAIP           | 20,984,171 |
| LINZ            | 3,299,006 |
| Sentinel-2-l2a | 18,683,945 |
| Landsat-c2l1    | 5,827,333 |
| Landsat-c2l2-sr | 5,790,651 |
| Sentinel-1-rtc | 16,133,394 |
| MODIS | 1350864|
=======
| NAIP           | 20984171 |
| LINZ            | 3299006 |
| Sentinel-2-l2a | 18683945 |
| Landsat-c2l1    | 5827333 |
| Landsat-c2l2-sr | 5790651 |
| Sentinel-1-rtc | 16133394 |
| MODIS          |  1350864 |
>>>>>>> main

# Older versions

For older versions of the model we used the following sampling stragegies.

## For model version v0.1

For v0.1 we used a smaller sample that was slightly less focused on human landscapes. The distribution of the MGRS tiles we used was as follows

| Class | Nr of Tiles | From highest |
|---|---|---|
Diversity | 500 | 3000
Built-up | 400 | 400
Herbaceous wetland | 50 | 500
Mangroves | 50 | 500
Moss and lichen | 50 | 500
Cropland | 100 | 500
Tree cover | 100 | 500
Shrubland | 50 | 500
Grassland | 50 | 500
Bare / sparse vegetation | 50 | 500
Snow and Ice | 50 | 500
Permanent water bodies | 100 | 1000

This resulted in a sample of 1517 MGRS tiles total in our sample.

The resulting sample file can be downloaded from the following link

https://clay-mgrs-samples.s3.amazonaws.com/mgrs_sample.fgb
