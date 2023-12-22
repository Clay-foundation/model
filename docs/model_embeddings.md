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
