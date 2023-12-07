import boto3

batch = boto3.client("batch", region_name="us-east-1")

NR_OF_TILES_IN_SAMPLE_FILE = 936

PC_KEY = "***"

for i in range(NR_OF_TILES_IN_SAMPLE_FILE):
    job = {
        "jobName": "fetch-and-run",
        "jobQueue": "fetch-and-run",
        "jobDefinition": "fetch-and-run",
        "containerOverrides": {
            "command": ["datacube.py", "--index", f"{i}", "--bucket", "clay-tiles-01"],
            "environment": [
                {"name": "BATCH_FILE_TYPE", "value": "zip"},
                {
                    "name": "BATCH_FILE_S3_URL",
                    "value": "s3://clay-fetch-and-run-zip/batch-fetch-and-run.zip",
                },
                {"name": "PC_SDK_SUBSCRIPTION_KEY", "value": f"{PC_KEY}"},
            ],
            "resourceRequirements": [
                {"type": "MEMORY", "value": "8000"},
                {"type": "VCPU", "value": "4"},
            ],
        },
    }

    print(batch.submit_job(**job))
