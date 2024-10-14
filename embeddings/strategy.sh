        # TODO: Remove for final
        with open("clay_embeddings.parquet", "wb") as dst:
            dst.write(body)
        import geopandas as gpd
        df = gpd.read_parquet("clay_embeddings.parquet")
        del df["embeddings"]
        df.crs = 4326
        df.to_file("clay_embeddings.gpkg")



    # with open("/Users/tam/Desktop/m_3008501_ne_16_1_20110815.tif") as f:


aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-2.amazonaws.com
docker pull 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2


aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 875815656045.dkr.ecr.us-east-1.amazonaws.com
docker build -t clay-v1-naip-embeddings .
docker tag clay-v1-naip-embeddings:latest 875815656045.dkr.ecr.us-east-1.amazonaws.com/clay-v1-naip-embeddings:latest
docker push 875815656045.dkr.ecr.us-east-1.amazonaws.com/clay-v1-naip-embeddings:latest


docker build -t clay-v1-naip-embeddings -f docker/Dockerfile .

docker run --rm -it \
  --cpus=6 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_BATCH_JOB_ARRAY_INDEX=0 \
  -e EMBEDDING_BATCH_SIZE=20 \
  -v /Users/tam/Desktop/m_4911964_sw_11_060_20210627.tif:/data/m_4911964_sw_11_060_20210627.tif \
  -v $PWD/docker/all-naip.py:/code/all-naip.py \
  clay-v1-naip-embeddings



        item = create_stac_item("/Users/tam/Desktop/m_4911964_sw_11_060_20210627.tif", with_proj=True)# TODO: remove!
MANIFEST = "/Users/tam/Documents/repos/model/data/naip-manifest.txt.zip"
CHECKPOINT = "/Users/tam/Documents/repos/model/data/checkpoints/clay-model-v1.5.0-september-30.ckpt"



        item = create_stac_item(
            "/Users/tam/Desktop/m_4911964_sw_11_060_20210627.tif", with_proj=True
        )  # TODO: remove!
