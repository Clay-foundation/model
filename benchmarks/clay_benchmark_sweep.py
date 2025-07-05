#!/usr/bin/env python3
"""
Comprehensive Clay Foundation Model Benchmark Sweep
Tests Clay across multiple PANGAEA datasets to demonstrate capabilities
"""

import subprocess
import time
import json
from datetime import datetime

# Configuration
DATASETS = [
    {
        "name": "hlsburnscars",
        "task": "segmentation",
        "epochs": 2,
        "batch_size": 8,
        "description": "Wildfire burn scar detection (binary, 6 optical bands)",
        "expected_miou": "65-75%",
    },
    {
        "name": "mados",
        "task": "segmentation",
        "epochs": 2,
        "batch_size": 8,
        "description": "Marine pollution detection (15-class, 11 optical bands)",
        "expected_miou": "15-25%",
    },
    {
        "name": "sen1floods11",
        "task": "segmentation",
        "epochs": 2,
        "batch_size": 4,
        "description": "Flood mapping (binary, SAR+Optical 15 bands)",
        "expected_miou": "70-80%",
    },
    {
        "name": "ai4smallfarms",
        "task": "segmentation",
        "epochs": 2,
        "batch_size": 8,
        "description": "Small farm mapping (binary, 4 optical bands)",
        "expected_miou": "75-85%",
    },
    {
        "name": "biomassters",
        "task": "regression",
        "epochs": 2,
        "batch_size": 8,
        "description": "Forest biomass estimation (regression, SAR+Optical)",
        "expected_miou": "MAE: 20-30",
    },
]


def run_benchmark(dataset_config):
    """Run Clay benchmark on a single dataset"""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {dataset_config['name'].upper()}")
    print(f"Description: {dataset_config['description']}")
    print(f"Expected: {dataset_config['expected_miou']}")
    print(f"{'='*60}")

    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=1",
        "pangaea/run.py",
        "--config-name=train",
        f'dataset={dataset_config["name"]}',
        "encoder=clay",
        f'task={dataset_config["task"]}',
        "use_wandb=false",
        f'task.trainer.n_epochs={dataset_config["epochs"]}',
        f'batch_size={dataset_config["batch_size"]}',
        "num_workers=4",
    ]

    # Add task-specific configurations
    if dataset_config["task"] == "segmentation":
        cmd.extend(
            [
                "decoder=seg_upernet",
                "preprocessing=seg_default",
                "criterion=cross_entropy",
            ]
        )
    elif dataset_config["task"] == "regression":
        cmd.extend(
            ["decoder=reg_upernet", "preprocessing=reg_default", "criterion=mse"]
        )

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )  # 30 min timeout
        elapsed = time.time() - start_time

        # Parse results from output
        success = result.returncode == 0
        return {
            "dataset": dataset_config["name"],
            "success": success,
            "elapsed_time": elapsed,
            "stdout": result.stdout[-2000:],  # Last 2000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "config": dataset_config,
        }
    except subprocess.TimeoutExpired:
        return {
            "dataset": dataset_config["name"],
            "success": False,
            "elapsed_time": 1800,
            "error": "Timeout after 30 minutes",
            "config": dataset_config,
        }
    except Exception as e:
        return {
            "dataset": dataset_config["name"],
            "success": False,
            "elapsed_time": time.time() - start_time,
            "error": str(e),
            "config": dataset_config,
        }


def main():
    """Run comprehensive benchmark sweep"""
    print("🚀 Starting Clay Foundation Model Benchmark Sweep")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Testing {len(DATASETS)} datasets")

    results = []

    for i, dataset_config in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Starting {dataset_config['name']}...")
        result = run_benchmark(dataset_config)
        results.append(result)

        # Quick status
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        elapsed_min = result["elapsed_time"] / 60
        print(
            f"[{i}/{len(DATASETS)}] {dataset_config['name']}: {status} ({elapsed_min:.1f}m)"
        )

    # Save results
    with open("clay_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate summary
    successful = sum(1 for r in results if r["success"])
    print("\n🏁 BENCHMARK COMPLETE")
    print(
        f"Success rate: {successful}/{len(DATASETS)} ({successful/len(DATASETS)*100:.1f}%)"
    )

    return results


if __name__ == "__main__":
    main()
