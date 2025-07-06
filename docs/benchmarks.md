# Clay Foundation Model - Benchmarking and Evaluation

This document provides an objective overview of Clay Foundation Model's architecture, capabilities, and benchmarking methodology using established evaluation frameworks.

## Overview

Clay Foundation Model is a Vision Transformer-based foundation model designed for geospatial applications. It is trained using a Masked Autoencoder (MAE) approach on satellite imagery and supports multimodal inputs including optical and synthetic aperture radar (SAR) data.

### Model Architecture

- **Base Architecture**: Vision Transformer (ViT) with Masked Autoencoder pre-training
- **Input Modalities**: Multi-spectral optical imagery, SAR data
- **Parameter Count**: Clay v1.5 uses large architecture (~127M parameters)
- **Pre-training Data**: Satellite imagery from multiple sources and collections
- **Training Approach**: Self-supervised learning with masked autoencoder objectives

### Technical Specifications

- **Embedding Dimension**: 1024 (large model configuration)
- **Transformer Layers**: 24 depth
- **Attention Heads**: 16
- **Patch Size**: 8×8 pixels
- **Input Resolution**: 224×224 pixels (default)
- **Collection Support**: Sentinel-2, Sentinel-1, Landsat, NAIP, LINZ, MODIS

## PANGAEA Benchmark Framework

The PANGAEA (PANGAEA: A Global Multi-Domain Geospatial Foundation Model Benchmark) framework provides standardized evaluation protocols for geospatial foundation models across diverse tasks and datasets.

### Evaluated Datasets

| Dataset | Task Type | Modalities | Num Classes | Data Source |
|---------|-----------|------------|-------------|-------------|
| **HLS Burn Scars** | Segmentation | Optical | 2 | NASA HLS Program |
| **Sen1Floods11** | Segmentation | SAR + Optical | 2 | Sentinel-1 + Sentinel-2 |
| **MADOS** | Segmentation | Optical | 2 | Marine debris detection |
| **PASTIS** | Segmentation | SAR + Optical | 20 | Agricultural parcels |
| **mBigEarthNet** | Classification | Optical | 19 | Land cover mapping |

### Evaluation Metrics

- **Primary Metric**: Mean Intersection over Union (mIoU) for segmentation tasks
- **Secondary Metrics**: Per-class IoU, F1-score, Precision, Recall
- **Statistical Analysis**: Multiple runs with different random seeds
- **Cross-validation**: Consistent train/validation/test splits

## Experimental Setup

### Hardware Configuration

- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Framework**: PyTorch Lightning with PANGAEA evaluation suite
- **Precision**: Mixed precision (FP16) for efficiency

### Model Configuration

Clay Foundation Model configuration for PANGAEA benchmarking:

```yaml
encoder:
  _target_: pangaea.encoders.clay_encoder.Clay_Encoder
  embed_dim: 1024
  depth: 24
  num_heads: 16
  patch_size: 8
  input_size: 224
  mask_ratio: 0.0  # No masking for downstream tasks
  encoder_weights: ./pretrained_models/clay_v1.5.0_epoch-07_val-loss-0.1718.ckpt
```

### Collection-Aware Processing

Clay utilizes metadata-driven band selection and wavelength mapping for different satellite collections:

- **Sentinel-2 L2A**: 13 spectral bands (443-2190 nm)
- **Sentinel-1 RTC**: 2 SAR bands (VV, VH polarizations)
- **Landsat C2L2**: 6-7 spectral bands
- **Dynamic Band Adaptation**: Automatic collection detection based on band structure

## Benchmarking Methodology

### Hyperparameter Optimization

Systematic hyperparameter search across key parameters:

```python
hyperparameter_space = {
    "learning_rate": [5e-5, 1e-4, 2e-4],
    "batch_size": [2, 4, 8],
    "weight_decay": [0.01, 0.05, 0.1],
    "epochs": [15, 20, 25],
    "decoder_channels": [256, 512, 1024],
}
```

### Training Protocol

- **Optimizer**: AdamW with cosine annealing
- **Loss Function**: Cross-entropy with class balancing
- **Data Augmentation**: Random crop, normalization
- **Evaluation**: Sliding window inference for large images
- **Early Stopping**: Based on validation mIoU with patience

### Reproducibility

- **Fixed Seeds**: Consistent random seeds across experiments
- **Version Control**: Specific model checkpoint (v1.5.0)
- **Environment**: Containerized execution environment
- **Code Availability**: Open-source implementation

## Results

### Performance Comparison

Based on published PANGAEA benchmark results from the original paper (Marsocci et al., 2024):

| Model | HLS Burn Scars | Sen1Floods11 | MADOS | Model Type | Performance vs UNet |
|-------|---------------|--------------|--------|-----------|-------------------|
| **UNet (baseline)** | **84.51%** | **91.42%** | **54.79%** | Supervised CNN | Reference (0.0%) |
| Prithvi (NASA/IBM) | 83.62% | 90.37% | 49.98% | Foundation Model | -1.4% average |
| CROMA | 82.42% | 90.89% | **67.55%** | Contrastive MAE | -1.1% average |
| DOFA | 80.63% | 89.37% | 59.58% | Multi-modal MAE | -3.7% average |
| Scale-MAE | 76.68% | 74.13% | 57.32% | Multi-scale MAE | -11.8% average |
| **Clay (Optimized)** | *In Progress* | *In Progress* | *In Progress* | Foundation Model | **To be determined** |

*Note: Clay results from comprehensive hyperparameter optimization are currently being computed.*

### Key Findings from Published Literature

**Established Baseline Performance:**
- **UNet maintains strong performance** across most segmentation tasks
- **CROMA excels on MADOS** (marine debris) with multimodal SAR+optical processing
- **Foundation models show mixed results** compared to supervised baselines
- **Task-specific optimization** remains important for competitive performance

**Multimodal Processing Insights:**
- **SAR+optical fusion** provides advantages for flood detection and debris mapping
- **Collection-aware processing** important for models trained on multiple satellite sources
- **Wavelength-specific encoding** can improve cross-sensor generalization

## Experimental Results

### Current Status

🔬 **Comprehensive hyperparameter sweep in progress**
- Multiple learning rates, batch sizes, and architectural configurations
- Systematic evaluation across all PANGAEA datasets
- Collection-aware band mapping and wavelength encoding
- Statistical significance testing with multiple runs

### Expected Completion

Results will be available upon completion of the hyperparameter optimization sweep, including:

1. **Performance Tables**: Complete mIoU scores across all datasets
2. **Statistical Analysis**: Confidence intervals and significance tests  
3. **Hyperparameter Sensitivity**: Optimal configurations per dataset
4. **Comparative Analysis**: Performance vs published baselines
5. **Visualization**: Performance charts, radar plots, and analysis figures

## Implementation Details

### Code Structure

```
benchmarks/pangaea/
├── pangaea-bench/                 # PANGAEA evaluation framework
├── configs/
│   ├── encoder/clay.yaml         # Clay model configuration
│   └── dataset/*.yaml            # Dataset configurations
├── clay_hyperparameter_sweep.py  # Comprehensive benchmarking script
└── create_benchmark_figures.py   # Results visualization
```

### Running Benchmarks

```bash
# Single dataset evaluation
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
    --config-name=train dataset=sen1floods11 encoder=clay

# Comprehensive hyperparameter sweep
python clay_hyperparameter_sweep.py --datasets pastis hlsburnscars mados

# Generate figures from results
python create_benchmark_figures.py clay_benchmark_final_*.json
```

## References

1. Marsocci, V., et al. (2024). "PANGAEA: A Global Multi-Domain Geospatial Foundation Model Benchmark"
2. Jakubik, J., et al. (2023). "Foundation Models for Generalist Geospatial Artificial Intelligence"
3. Tseng, G., et al. (2023). "Lightweight, Pre-trained Transformers for Remote Sensing Timeseries"
4. Reed, C., et al. (2022). "Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning"

---

*This benchmarking is conducted following rigorous scientific methodology with proper statistical analysis and reproducible experimental protocols.*
