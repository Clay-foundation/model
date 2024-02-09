## Upload package to fetch and run bucket

```bash
zip -FSr batch-fetch-and-run-wc.zip src scripts -x *.pyc -x scripts/worldcover/wandb/**\*
# Add run to home dir
zip -uj batch-fetch-and-run-wc.zip scripts/worldcover/run.py

aws s3api put-object --bucket clay-fetch-and-run-packages --key "batch-fetch-and-run-wc.zip" --body "batch-fetch-and-run-wc.zip"
```

# Push array job

```python
import boto3

batch = boto3.client("batch", region_name="us-east-1")
job = {
    "jobName": f"fetch-and-run",
    "jobQueue": "fetch-and-run",
    "jobDefinition": "fetch-and-run",
    "containerOverrides": {
        "command": ["run.py"],
        "environment": [
            {"name": "BATCH_FILE_TYPE", "value": "zip"},
            {
                "name": "BATCH_FILE_S3_URL",
                "value": "s3://clay-fetch-and-run-packages/batch-fetch-and-run-wc.zip",
            }
        ],
        "resourceRequirements": [
            {"type": "MEMORY", "value": "8000"},
            {"type": "VCPU", "value": "4"},
            # {"type": "GPU", "value": "1"},
        ],
    },
    "arrayProperties": {
        # "size": int((125 - 67) * 12000 / 512)
        "size": 3
    },
    "retryStrategy": {
      "attempts": 5,
      "evaluateOnExit": [
        {"onStatusReason": "Host EC2*", "action": "RETRY"},
        {"onReason": "*", "action": "EXIT"}
      ]
    },
}

print(batch.submit_job(**job))
```
