# Generating vector embeddings

Once you have a pretrained model, it is now possible to pass some input images
into the encoder part of the Vision Transformer, and produce vector embeddings
which contain a semantic representation of the image.

## Producing embeddings from the pretrained model

Step by step instructions to create embeddings for a single MGRS tile location
(e.g. 27WXN).

1. Ensure that you can access the 13-band GeoTIFF data files.

   ```
   aws s3 ls s3://clay-tiles-02/02/27WXN/
   ```

   This should report a list of filepaths if you have the correct permissions,
   otherwise, please set up authentication before continuing.

2. Download the pretrained model weights, and put them in the `checkpoints/`
   folder.

   ```bash
   aws s3 cp s3://clay-model-ckpt/v0/clay-small-70MT-1100T-10E.ckpt checkpoints/
   ```

   ```{tip}
   For running model inference on a large scale (hundreds or thousands of MGRS
   tiles), it is recommended to have a cloud VM instance with:

   1. A high bandwidth network (>25Gbps) to speed up data transfer from the S3
      bucket to the compute device.
   2. An NVIDIA Ampere generation GPU (e.g. A10G) or newer, which would allow
      for efficient bfloat16 dtype calculations.

   For example, an AWS g5.4xlarge instance would be a cost effective option.
   ```

3. Run model inference to generate the embeddings.

   ```bash
   python trainer.py predict --ckpt_path=checkpoints/clay-small-70MT-1100T-10E.ckpt \
                             --trainer.precision=bf16-mixed \
                             --data.data_dir=s3://clay-tiles-02/02/27WXN \
                             --data.batch_size=32 \
                             --data.num_workers=16
   ```

   This should output a GeoParquet file containing the embeddings for MGRS tile
   27WXN (recall that each 10000x10000 pixel MGRS tile contains hundreds of
   smaller 512x512 chips), saved to the `data/embeddings/` folder. See the next
   sub-section for details about the embeddings file.

   ```{note}
   For those interested in how the embeddings were computed, the predict step
   above does the following:

   1. Pass the 13-band GeoTIFF input into the Vision Transformer's encoder, to
      produce raw embeddings of shape (B, 1538, 768), where B is the batch_size,
      1538 is the patch dimension and 768 is the embedding length. The patch
      dimension itself is a concatenation of 1536 (6 band groups x 16x16
      spatial patches of size 32x32 pixels each in a 512x512 image) + 2 (latlon
      embedding and time embedding) = 1538.
   2. The mean or average is taken across the 1536 patch dimension, yielding an
      output embedding of shape (B, 768).

   More details of how this is implemented can be found by inspecting the
   `predict_step` method in the `model_clay.py` file.
   ```


## Format of the embedding file

The vector embeddings are stored in a single column within a
[GeoParquet](https://geoparquet.org) file (*.gpq), with other columns
containing spatiotemporal metadata. This file format is built on top of the
popular Apache Parquet columnar storage format designed for fast analytics,
and it is highly interoperable across different tools like QGIS,
GeoPandas (Python), sfarrow (R), and more.

### Filename convention

The embeddings file utilizes the following naming convention:

```
{MGRS:5}_{MINDATE:8}_{MAXDATE:8}_v{VERSION:3}.gpq
```

Example: `27WXN_20200101_20231231_v001.gpq`

| Variable | Description |
|--|--|
| MGRS | The spatial location of the file's contents in the [Military Grid Reference System (MGRS)](https://en.wikipedia.org/wiki/Military_Grid_Reference_System), given as a 5-character string |
| MINDATE | The minimum acquisition date of the Sentinel-2 images used to generate the embeddings, given in YYYYMMDD format |
| MINDATE | The maximum acquisition date of the Sentinel-2 images used to generate the embeddings, given in YYYYMMDD format |
| VERSION | Version of the generated embeddings, given as a 3-digit number |


### Table schema

Each row within the GeoParquet table is generated from a 512x512 pixel image,
and contains a record of the embeddings, spatiotemporal metadata, and a link to
the GeoTIFF file used as the source image for the embedding. The table looks
something like this:

|         source_url          |    date    |      embeddings      |   geometry   |
|-----------------------------|------------|----------------------|--------------|
| s3://.../.../claytile_*.tif | 2021-01-01 | [0.1, 0.4, ... x768] | POLYGON(...) |
| s3://.../.../claytile_*.tif | 2021-06-30 | [0.2, 0.5, ... x768] | POLYGON(...) |
| s3://.../.../claytile_*.tif | 2021-12-31 | [0.3, 0.6, ... x768] | POLYGON(...) |

Details of each column are as follows:

- `source_url` ([string](https://arrow.apache.org/docs/python/generated/pyarrow.string.html)) - The full URL to the 13-band GeoTIFF image the embeddings were derived from.
- `date` ([date32](https://arrow.apache.org/docs/python/generated/pyarrow.date32.html)) - Acquisition date of the Sentinel-2 image used to generate the embeddings, in YYYY-MM-DD format.
- `embeddings` ([FixedShapeTensorArray](https://arrow.apache.org/docs/python/generated/pyarrow.FixedShapeTensorArray.html)) - The vector embeddings given as a 1-D tensor or list with a length of 768.
- `geometry` ([binary](https://arrow.apache.org/docs/python/generated/pyarrow.binary.html)) - The spatial bounding box of where the 13-band image, provided in a [WKB](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry#Well-known_binary) Polygon representation.


```{note}
Additional technical details of the GeoParquet file:
- GeoParquet specification [v1.0.0](https://geoparquet.org/releases/v1.0.0)
- Coordinate reference system of geometries are in `OGC:CRS84`.
```

## Reading the embeddings

Sample code to read the GeoParquet embeddings file using
[`geopandas.read_parquet`](https://geopandas.org/en/stable/docs/reference/api/geopandas.read_parquet.html)

```{code}
import geopandas as gpd

gpq_file = "data/embeddings/27WXN_20200101_20231231_v001.gpq"
geodataframe = gpd.read_parquet(path=gpq_file)
print(geodataframe)
```

```{seealso}
Further reading:
- https://guide.cloudnativegeo.org/geoparquet
- https://cloudnativegeo.org/blog/2023/10/the-geoparquet-ecosystem-at-1.0.0
```
