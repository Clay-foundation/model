import os

import boto3

batch = boto3.client("batch", region_name="us-east-1")

NR_OF_TILES_IN_SAMPLE_FILE = 2728

PC_KEY = os.environ["PC_SDK_SUBSCRIPTION_KEY"]


job = {
    "jobName": "fetch-and-run-datacube",
    "jobQueue": "fetch-and-run",
    "jobDefinition": "fetch-and-run",
    "containerOverrides": {
        "command": [
            "datacube.py",
            "--bucket",
            "clay-tiles-04-sample-v02",
            "--sample",
            "https://clay-mgrs-samples.s3.amazonaws.com/mgrs_sample_v02.fgb",
        ],
        "environment": [
            {"name": "BATCH_FILE_TYPE", "value": "zip"},
            {
                "name": "BATCH_FILE_S3_URL",
                "value": "s3://clay-fetch-and-run-packages/batch-fetch-and-run.zip",
            },
            {"name": "PC_SDK_SUBSCRIPTION_KEY", "value": f"{PC_KEY}"},
        ],
        "resourceRequirements": [
            {"type": "MEMORY", "value": "15500"},
            {"type": "VCPU", "value": "4"},
        ],
    },
    "arrayProperties": {"size": NR_OF_TILES_IN_SAMPLE_FILE},
}

print(batch.submit_job(**job))
