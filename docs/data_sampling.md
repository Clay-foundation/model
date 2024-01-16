# Data sampling strategy

To create a balanced dataset for model training, we used a sampling strategy
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

The following table summarizes the selection criteria for each class.

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

After selecting MGRS tiles for each of these criteria, we removed duplicates.
This resulted in a sample of 1517 MGRS tiles total in our sample.

The resulting sample file can be downloaded from the following link

https://clay-mgrs-samples.s3.amazonaws.com/mgrs_sample.fgb
