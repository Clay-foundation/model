FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

WORKDIR /model

RUN git clone -b all-of-naip https://github.com/Clay-foundation/model.git .

RUN aws s3 cp --no-sign-request s3://clay-model-ckpt/v1.5.0-no-mrl-dinov2/mae_v1.5.0_epoch-07_val-loss-0.1718.ckpt data/mae_v1.5.0_epoch-07_val-loss-0.1718.ckpt
RUN aws s3 cp --no-sign-request s3://clay-mgrs-samples/naip-manifest.txt.zip data/naip-manifest.txt.zip
COPY embeddings/data/stac_manifest_brazil_sentinel_2023_2024.csv.gz data/

RUN pip install \
  einops~=0.7.0 \
  fiona~=1.9.5 \
  geopandas~=0.14.1 \
  jsonargparse~=4.27.0 \
  lightning~=2.1.0 \
  matplotlib~=3.9.0 \
  planetary-computer~=1.0.0 \
  python-box~=7.1.0 \
  pyarrow~=15.0.2 \
  rasterio~=1.3.10 \
  s3fs~=2024.6.0 \
  boto3~=1.34.122 \
  botocore~=1.34.122 \
  scikit-image~=0.22.0 \
  scikit-learn~=1.4.0 \
  stackstac~=0.5.0 \
  timm~=0.9.16 \
  transformers~=4.35.2 \
  typeshed-client~=2.4.0 \
  vit-pytorch~=1.6.4 \
  zarr~=2.16.1 \
  geoarrow-pyarrow==0.1.2 \
  torchdata==0.7.1 \
  stacchip==0.1.35 \
  wandb==0.17.5 \
  rio_stac~=0.10.0

RUN git pull && git checkout ceecb6138705cb28a5f4d3f61f22b19a2f625edb

COPY embeddings/brazil-23-24-sentinel2.py .

ENTRYPOINT ["python"]
