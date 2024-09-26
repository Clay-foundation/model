FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

RUN pip install stacchip

WORKDIR /src

ADD data data
ADD checkpoints checkpoints
ADD all-naip.py .

ENTRYPOINT [ "python", "all-naip.py" ]
