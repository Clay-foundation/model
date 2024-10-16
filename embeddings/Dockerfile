FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

WORKDIR /model

RUN aws s3 cp s3://clay-model-ckpt/v1.5.0/mae_v1.5.0_epoch-76_val-loss-0.1612.ckpt data/clay-model-v1.5.0-october-12.ckpt
RUN aws s3 cp s3://clay-mgrs-samples/naip-manifest.txt.zip data/naip-manifest.txt.zip

RUN pip install \
  einops~=0.7.0 \
  fiona~=1.9.5 \
  geopandas~=0.14.1 \
  jsonargparse~=4.27.0 \
  lightning~=2.1.0 \
  matplotlib~=3.8.2 \
  planetary-computer~=1.0.0 \
  python-box~=7.1.0 \
  pyarrow~=16.1.0 \
  rasterio~=1.3.10 \
  s3fs~=2024.3.1 \
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
  wandb==0.17.5

RUN git clone -b all-of-naip https://github.com/Clay-foundation/model.git .
# Move file to home directory so that relative imports work
RUN cp embeddings/all-naip.py .
RUN cp embeddings/all-sentinel.py .

ENTRYPOINT ["python", "all-naip.py"]
