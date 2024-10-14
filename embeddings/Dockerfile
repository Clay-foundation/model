FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

WORKDIR /model

RUN aws s3 cp s3://clay-model-ckpt/v1.5.0/mae_v1.5.0_epoch-76_val-loss-0.1612.ckpt /data/clay-model-v1.5.0-october-12.ckpt
RUN aws s3 cp s3://clay-mgrs-samples/naip-manifest.txt.zip /data/naip-manifest.txt.zip

RUN git clone -b all-of-naip https://github.com/Clay-foundation/model.git /model

# ADD docker/environment.yml docker-environment.yml
RUN mamba env create --file environment.yml

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "claymodel", "python", "docker/all-naip.py"]
