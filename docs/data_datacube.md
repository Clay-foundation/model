# Creating datacubes

## How to create a datacube

TODO:
- Describe how the `datacube.py` script works for a single MGRS tile

## Running the datacube pipeline as a batch job

How to run the data pipeline on AWS Batch Spot instances using
a [fetch-and-run](https://aws.amazon.com/blogs/compute/creating-a-simple-fetch-and-run-aws-batch-job/)
approach.

### Prepare docker image in ECR

Build the docker image and push it to a ecr repository.

```bash
ecr_repo_id=12345
cd batch
docker build -t $ecr_repo_iud.dkr.ecr.us-east-1.amazonaws.com/fetch-and-run .

aws ecr get-login-password --profile clay --region us-east-1 | docker login --username AWS --password-stdin $ecr_repo_iud.dkr.ecr.us-east-1.amazonaws.com

docker push $ecr_repo_iud.dkr.ecr.us-east-1.amazonaws.com/fetch-and-run:latest
```

### Prepare AWS batch

To prepare batch, we need to create a compute environment, job queue, and job
definition.

Example configurations for the compute environment and the job definition are
provided in the `batch` directory.

The `submit.py` script contains a loop for submitting jobs to the queue. An
alternative to this individual job submissions would be to use array jobs, but
for now the individual submissions are simpler and failures are easier to track.

### Create ZIP file with the package to execute

Package the model and the inference script into a zip file. The `datacube.py`
script is the one that will be executed on the instances.

Put the scripts in a zip file and upload the zip package into S3 so that
the batch fetch and run can use it.

```bash
zip -FSrj "batch-fetch-and-run.zip" ./scripts/* -x "scripts/*.pyc"

aws s3api put-object --bucket clay-fetch-and-run-packages --key "batch-fetch-and-run.zip" --body "batch-fetch-and-run.zip"
```

### Submit job

We can now submit a batch job to run the pipeline. The `submit.py` file
provides an example on how to sumbit jobs in python.
