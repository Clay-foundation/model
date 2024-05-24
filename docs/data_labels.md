# Benchmark dataset labels

A benchmark dataset is a collection of data used for evaluating the performance
of algorithms, models, or systems in a specific field of study. These datasets
are crucial for providing common ground for comparing different approaches,
allowing researchers to assess the strengths and weaknesses of various methods.
For Clay, we evaluate our model on benchmark datasets that have suitable downstream
tasks.

For our initial benchmark dataset, we've implemented the
[Cloud to Street - Microsoft flood dataset](https://beta.source.coop/repositories/c2sms/c2smsfloods/description).
It is what we will use in our initial linear probing experiments and
evaluation of finetuning on a downstream task. The task itself is
[segmentation](https://paperswithcode.com/task/semantic-segmentation) of water
pixels associated with recorded flood events.

The original dataset consists of two out of three of our Foundation model's datacube inputs
(Sentinel-1 and Sentinel-2) along with raster water mask labels for both
sensors. Each image is 512x512 pixels. The
original Sentinel-2 images are L1C, which is Top-of-Atmosphere reflectance. We train
Clay with surface reflectance, however, so we ultimately used the geospatial bounds
from the GeoTIFF and image timestamp (from the granule name) to query
[Microsoft Planetary Computer's STAC API for L2A (Bottom-of-Atmosphere a.k.a. "surface reflectance") Sentinel-2](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)
scenes in the same time and space, with the same channels expected by Clay. We
then followed the same `datacube` creation logic to generate datacubes with
Sentinel-1 VV and VH and the Copernicus Digital Elevation Model (DEM). We also
ensured that the Sentinel-1 data was within a +/- 3 day interval of each
reference Sentinel-2 scene (same method used by the benchmark dataset authors)
and that the Sentinel-1 data was indeed already included in the bechmark
dataset's list of granules. The datacubes generated have all three inputs
matching the exact specs of the Foundation model's training data, at 512x512
pixels.

Here is an example of a datacube we generated for the dataset:

![datacube](https://github.com/Clay-foundation/model/assets/23487320/94dffcf5-4075-4c17-ac96-01c11bcb299b)

The images, left to right, show a true-color representation of the Sentinel-2
scene, the Sentinel-1 VH polarization, and the Digital Elevation Model.

![gt](https://github.com/Clay-foundation/model/assets/23487320/4ac92af7-6931-4249-a920-7d29453b9b31)

Here we have something similar, but this time just the Sentinel-1 and
Sentinel-2 scenes with the Sentinel-1 water mask (ground truth) overlaid.

Last note on this benchmark dataset that we've adapted for Clay: we made sure
to preserve the metadata for timestamp and geospatial coordinates in the
datacube such that we can embed information in the way that the Clay Foundation
model expects. We also preserve the flood event information for analysis
during finetuning.

The script for generating these datacubes is at
https://github.com/Clay-foundation/model/blob/c2smsfloods_benchmark_datapipeline/scripts/datacube_benchmark.py.
You'll need an AWS account and Microsoft Planetary Computer API Key to run
this. The data is queried from Microsoft Planetary Computer STAC APIs, read and
processed in memory, and the datacubes are written directly to AWS S3.
