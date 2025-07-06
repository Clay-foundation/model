# Clay PANGAEA Benchmark Integration

This directory contains the integration of Clay Foundation Model with the PANGAEA benchmark suite for comprehensive evaluation on geospatial tasks.

## Overview

Clay Foundation Model is evaluated on the PANGAEA (PANGAEA: A Global Multi-Domain Geospatial Foundation Model Benchmark) datasets to assess its performance across diverse Earth observation tasks.

## Files

- `run_clay_benchmark.py` - Main benchmarking script for Clay model evaluation
- `analyze_clay_results.py` - Results analysis and visualization tools
- `results/` - Directory containing sample benchmark results
- `pangaea-bench/` - PANGAEA framework submodule

## Usage

### Running Benchmarks

```bash
# Run benchmark on all datasets
python run_clay_benchmark.py

# Run on specific datasets
python run_clay_benchmark.py --datasets hlsburnscars sen1floods11 mados pastis mbigearthnet

# Customize number of configurations per dataset
python run_clay_benchmark.py --max-configs 8
```

### Analyzing Results

```bash
# Analyze results and generate report
python analyze_clay_results.py results/clay_pangaea_results_*.json

# Specify output directory
python analyze_clay_results.py results/clay_pangaea_results_*.json --output analysis_output
```

## Datasets

The benchmark evaluates Clay on these PANGAEA datasets:

| Dataset | Task | Bands | Description |
|---------|------|-------|-------------|
| HLS Burn Scars | Segmentation | 6 | Wildfire burn scar detection |
| Sen1Floods11 | Segmentation | 15 | Flood detection (SAR+optical) |
| MADOS | Segmentation | 11 | Marine debris detection |
| PASTIS | Segmentation | 10 | Agricultural crop segmentation |
| M-BigEarthNet | Classification | 12 | Land cover classification |

## Setup

1. **Install PANGAEA benchmark dependencies:**
   ```bash
   cd pangaea-bench
   pip install -r requirements.txt --break-system-packages
   pip install --no-build-isolation --no-deps -e . --break-system-packages
   ```

2. **Clay model checkpoint:**
   The Clay checkpoint is automatically copied to `pangaea-bench/pretrained_models/clay_v1.5.0_epoch-07_val-loss-0.1718.ckpt`

## Manual Commands

#### Semantic Segmentation with MADOS dataset:
```bash
cd pangaea-bench
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=mados \
   encoder=clay \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   use_wandb=False
```

## Clay Encoder Configuration

The Clay encoder configuration is located at `pangaea-bench/configs/encoder/clay.yaml` and includes:

- **Model**: Clay Foundation Model v1.5.0
- **Architecture**: Vision Transformer with DOFA (Do One For All) design
- **Embed dimension**: 1024 (large model)
- **Layers**: 24 depth, 16 attention heads
- **Patch size**: 8x8 pixels

## Key Features

- **Automated hyperparameter optimization** - Smart search across optimal configurations
- **Multi-modal support** - Handles both optical and SAR data
- **Comprehensive analysis** - Performance comparison with published baselines
- **Visualization tools** - Automated generation of figures and reports

## Results

The analysis generates:
- Performance comparison charts vs PANGAEA baselines
- Hyperparameter sensitivity analysis
- Comprehensive markdown reports
- CSV summaries for further analysis

For detailed results and methodology, see `docs/benchmarks.md`.
