## Large scale embedding runs

The code in this section has been used to create embedding runs over large
archives. Currently this covers NAIP and Sentinel-2.

The algorithms are dockerized to be ran in a batch setup. AWS Batch is what
was used to execute the algorithms but it is not a strict requirement.

The scripts rely on the `AWS_BATCH_JOB_ARRAY_INDEX` environment variable
to choose which files from the archives to process. This is set automatically
by AWS Batch when using array jobs. Outside of array jobs, this index variable
needs to be specified manually.

### Build docker image

Embedding runs are dockerized for parallel computing. To build the docker image
use the Dockerfile in the embeddings directory. Then push the image to ECR or
another docker repository of your choice.

```bash
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-2.amazonaws.com
docker pull 763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

docker build -t clay-embeddings -f embeddings/Dockerfile  .

docker tag clay-embeddings:latest 875815656045.dkr.ecr.us-east-2.amazonaws.com/clay-embeddings:latest
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 875815656045.dkr.ecr.us-east-2.amazonaws.com
docker push 875815656045.dkr.ecr.us-east-2.amazonaws.com/clay-embeddings:latest
```

### NAIP

For NAIP, we use the `naip-analytic` bucket. We leverage the manifest file that
lists all files in the bucket. This list is parsed in the beginning and each
job processes a section of the naip scenes.

At the moment of processing there were 1'231'441 NAIP scenes.

### Sentinel-2

For Sentinel-2 we use the `sentinel-cogs` bucket. Also here we use the manifest
file, but parse it beforehand because it contains references to each single
asset for each product.

The parser is essentially copied from [this gist](https://github.com/alexgleith/sinergise-element84-sentinel-2-qa/blob/main/0-parse-inventory-element84.py)
by @alexgleith.
The resulting zip file contains a list of static STAC json files for 2023 and 2024.
