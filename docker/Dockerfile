FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

WORKDIR /code

ADD src src

ADD docker/environment.yml docker-environment.yml
RUN mamba env create --file docker-environment.yml

ADD data/naip-manifest.txt.zip /data/naip-manifest.txt.zip
ADD data/checkpoints/clay-model-v1.5.0-september-30.ckpt /data/clay-model-v1.5.0-september-30.ckpt

ADD docker/all-naip.py .
ADD configs configs

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "claymodel", "python", "all-naip.py"]
