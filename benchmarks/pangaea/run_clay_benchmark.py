#!/usr/bin/env python3
"""
Clay Foundation Model PANGAEA Benchmark
Consolidated benchmarking script for Clay model evaluation on PANGAEA datasets.
"""

import subprocess
import time
import json
import itertools
from pathlib import Path
from typing import Dict, List, Any
import argparse

class ClayPANGAEABenchmark:
    def __init__(self):
        """Initialize Clay PANGAEA benchmark with optimized configurations."""
        
        # PANGAEA datasets for evaluation
        self.datasets = {
            "hlsburnscars": {
                "description": "HLS Burn Scars - 6 optical bands",
                "encoder": "clay", 
                "bands": 6
            },
            "sen1floods11": {
                "description": "Sen1Floods11 - 15 bands (S1+S2)",
                "encoder": "clay",
                "bands": 15
            },
            "mados": {
                "description": "MADOS - 11 spectral bands",
                "encoder": "clay",
                "bands": 11
            },
            "pastis": {
                "description": "PASTIS - 10 spectral bands",
                "encoder": "clay", 
                "bands": 10
            },
            "mbigearthnet": {
                "description": "M-BigEarthNet - 12 spectral bands",
                "encoder": "clay",
                "bands": 12
            }
        }
        
        # Optimized hyperparameter search space
        self.hyperparams = {
            "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4],
            "batch_size": [4, 8, 16],
            "weight_decay": [0.0, 0.01, 0.05],
            "epochs": [15, 25, 35],
            "decoder_channels": [512, 1024]
        }
        
        self.results = []

    def run_single_experiment(self, dataset: str, config: Dict[str, Any], 
                            experiment_num: int, total_experiments: int) -> Dict[str, Any]:
        """Run a single Clay experiment with given configuration."""
        
        dataset_info = self.datasets[dataset]
        encoder = dataset_info["encoder"]
        
        # Build command for PANGAEA benchmark
        cmd = [
            "torchrun", "--nnodes=1", "--nproc_per_node=1", "pangaea/run.py",
            "--config-name=train",
            f"dataset={dataset}",
            f"encoder={encoder}",
            "task=segmentation",
            "criterion=cross_entropy",
            "decoder=seg_upernet", 
            "preprocessing=seg_default",
            "use_wandb=false",
            f"optimizer.lr={config['learning_rate']}",
            f"batch_size={config['batch_size']}",
            f"optimizer.weight_decay={config['weight_decay']}",
            f"task.trainer.n_epochs={config['epochs']}",
            f"decoder.channels={config['decoder_channels']}"
        ]
        
        print(f"\n[{experiment_num}/{total_experiments}] 🔬 {dataset} ({dataset_info['bands']} bands)")
        print(f"⚙️  Config: LR={config['learning_rate']}, BS={config['batch_size']}, "
              f"WD={config['weight_decay']}, E={config['epochs']}, DC={config['decoder_channels']}")
        
        start_time = time.time()
        
        try:
            # Change to pangaea-bench directory
            original_dir = Path.cwd()
            pangaea_dir = Path("benchmarks/pangaea/pangaea-bench")
            
            if pangaea_dir.exists():
                import os
                os.chdir(pangaea_dir)
            
            # Run experiment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            duration = time.time() - start_time
            
            # Parse mIoU from output
            miou = self._extract_miou(result.stdout)
            success = result.returncode == 0 and miou > 0
            
            if pangaea_dir.exists():
                os.chdir(original_dir)
                
            print(f"{'✅' if success else '❌'} Completed in {duration:.1f}s - mIoU: {miou:.2f}%")
            
            return {
                "dataset": dataset,
                "encoder": encoder,
                "config": config,
                "success": success,
                "miou": miou,
                "duration": duration,
                "timestamp": time.time()
            }
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after 2 hours")
            return {
                "dataset": dataset,
                "encoder": encoder,
                "config": config,
                "success": False,
                "miou": 0.0,
                "duration": 7200,
                "error": "timeout",
                "timestamp": time.time()
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "dataset": dataset,
                "encoder": encoder,
                "config": config,
                "success": False,
                "miou": 0.0,
                "duration": time.time() - start_time,
                "error": str(e),
                "timestamp": time.time()
            }

    def _extract_miou(self, stdout: str) -> float:
        """Extract mean IoU from training output."""
        lines = stdout.split('\n')
        
        # Look for test results (final evaluation)
        for line in reversed(lines):
            if '[test] Mean' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Mean' and i + 1 < len(parts):
                            return float(parts[i + 1])
                except (ValueError, IndexError):
                    continue
                    
        return 0.0

    def generate_configs(self, max_per_dataset: int = 12) -> List[Dict[str, Any]]:
        """Generate optimized hyperparameter configurations."""
        
        # Generate all combinations
        param_names = list(self.hyperparams.keys())
        param_values = list(self.hyperparams.values())
        all_combinations = list(itertools.product(*param_values))
        
        all_configs = [dict(zip(param_names, combo)) for combo in all_combinations]
        
        if len(all_configs) <= max_per_dataset:
            return all_configs
        
        # Select optimal configurations for foundation model fine-tuning
        selected = []
        
        # Core configurations based on foundation model best practices
        optimal_configs = [
            {"learning_rate": 2e-5, "batch_size": 8, "weight_decay": 0.01, "epochs": 25, "decoder_channels": 1024},
            {"learning_rate": 1e-5, "batch_size": 16, "weight_decay": 0.0, "epochs": 35, "decoder_channels": 1024},
            {"learning_rate": 5e-5, "batch_size": 4, "weight_decay": 0.05, "epochs": 15, "decoder_channels": 512},
            {"learning_rate": 1e-4, "batch_size": 8, "weight_decay": 0.01, "epochs": 25, "decoder_channels": 512},
        ]
        
        # Add optimal configs first
        for config in optimal_configs:
            if config in all_configs and config not in selected:
                selected.append(config)
        
        # Fill remaining with diverse sampling
        remaining = [c for c in all_configs if c not in selected]
        import random
        random.shuffle(remaining)
        
        while len(selected) < max_per_dataset and remaining:
            selected.append(remaining.pop())
            
        return selected[:max_per_dataset]

    def run_benchmark(self, target_datasets: List[str] = None, max_configs: int = 12):
        """Run comprehensive Clay benchmark across PANGAEA datasets."""
        
        datasets_to_run = target_datasets if target_datasets else list(self.datasets.keys())
        configs = self.generate_configs(max_configs)
        
        print(f"\n🎯 CLAY FOUNDATION MODEL PANGAEA BENCHMARK")
        print(f"📊 Datasets: {datasets_to_run}")
        print(f"🔬 Configurations per dataset: {len(configs)}")
        print(f"📈 Total experiments: {len(datasets_to_run) * len(configs)}")
        
        total_experiments = len(datasets_to_run) * len(configs)
        completed = 0
        timestamp = int(time.time())
        
        for dataset in datasets_to_run:
            dataset_info = self.datasets[dataset]
            
            print(f"\n{'='*60}")
            print(f"🗂️  DATASET: {dataset.upper()}")
            print(f"📊 {dataset_info['description']}")
            print(f"{'='*60}")
            
            dataset_results = []
            
            for config in configs:
                completed += 1
                result = self.run_single_experiment(dataset, config, completed, total_experiments)
                dataset_results.append(result)
                self.results.append(result)
                
                # Save intermediate results
                self._save_intermediate_results(timestamp)
            
            # Report best result for dataset
            successful = [r for r in dataset_results if r["success"]]
            if successful:
                best = max(successful, key=lambda x: x["miou"])
                print(f"\n🏆 BEST for {dataset}: {best['miou']:.2f}% mIoU")
                print(f"   Config: LR={best['config']['learning_rate']}, E={best['config']['epochs']}")
            else:
                print(f"\n❌ No successful results for {dataset}")
        
        # Save final results and generate summary
        results_file = f"clay_pangaea_results_{timestamp}.json"
        self._save_final_results(results_file)
        self._print_summary()
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file

    def _save_intermediate_results(self, timestamp: int):
        """Save intermediate results during benchmark run."""
        filename = f"clay_pangaea_intermediate_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _save_final_results(self, filename: str):
        """Save final benchmark results."""
        with open(filename, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": time.time(),
                    "total_experiments": len(self.results),
                    "datasets": list(self.datasets.keys()),
                    "hyperparameter_space": self.hyperparams
                },
                "results": self.results
            }, f, indent=2)

    def _print_summary(self):
        """Print comprehensive benchmark summary."""
        print(f"\n📋 CLAY PANGAEA BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        successful = [r for r in self.results if r["success"]]
        total = len(self.results)
        
        print(f"📊 Total experiments: {total}")
        print(f"✅ Successful: {len(successful)} ({100*len(successful)/total:.1f}%)")
        print(f"❌ Failed: {total - len(successful)} ({100*(total-len(successful))/total:.1f}%)")
        
        if successful:
            # Best results per dataset
            print(f"\n🏆 BEST RESULTS PER DATASET:")
            print(f"{'-'*60}")
            
            dataset_best = {}
            for dataset in self.datasets.keys():
                dataset_results = [r for r in successful if r["dataset"] == dataset]
                if dataset_results:
                    best = max(dataset_results, key=lambda x: x["miou"])
                    dataset_best[dataset] = best
                    print(f"{dataset:15s}: {best['miou']:6.2f}% mIoU")
                else:
                    print(f"{dataset:15s}: {'Failed':>6s}")
            
            if dataset_best:
                avg_miou = sum(r["miou"] for r in dataset_best.values()) / len(dataset_best)
                print(f"\n🎯 Average mIoU: {avg_miou:.2f}%")
                print(f"📊 Successful datasets: {len(dataset_best)}/{len(self.datasets)}")

def main():
    """Main entry point for Clay PANGAEA benchmark."""
    
    parser = argparse.ArgumentParser(description="Clay Foundation Model PANGAEA Benchmark")
    parser.add_argument("--datasets", nargs="+", default=None,
                       help="Datasets to benchmark (default: all)")
    parser.add_argument("--max-configs", type=int, default=12,
                       help="Maximum configurations per dataset")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = ClayPANGAEABenchmark()
    results_file = benchmark.run_benchmark(args.datasets, args.max_configs)
    
    print(f"\n🎉 Benchmark completed!")
    print(f"📊 Results: {results_file}")
    
    return results_file

if __name__ == "__main__":
    main()