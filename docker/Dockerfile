FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

RUN pip install stacchip

WORKDIR /src

ADD docker/data data
ADD docker/checkpoints checkpoints
ADD docker/all-naip.py .

ADD environment.yml environment.yml
RUN conda install -y environment.yml

ENTRYPOINT [ "python", "all-naip.py" ]
