# Clay Foundation Model - Benchmarking and Evaluation

This document provides an objective overview of Clay Foundation Model's architecture, capabilities, and benchmarking methodology using established evaluation frameworks.

## Overview

Clay Foundation Model is a Vision Transformer-based foundation model designed for geospatial applications. It is trained using a Masked Autoencoder (MAE) approach on satellite imagery and supports multimodal inputs including optical and synthetic aperture radar (SAR) data.

### Model Architecture

- **Base Architecture**: Vision Transformer (ViT) with Masked Autoencoder pre-training
- **Input Modalities**: Multi-spectral optical imagery, SAR data
- **Parameter Count**: Approximately 100M+ parameters
- **Pre-training Data**: Satellite imagery from multiple sources and sensors
- **Patch Size**: 16x16 pixels (configurable)
- **Supported Bands**: Variable, typically 4-15 spectral bands

### Key Capabilities

- **Multimodal Processing**: Native support for both optical and SAR data
- **Variable Input Size**: Flexible handling of different image dimensions
- **Band Flexibility**: Adaptable to different numbers of spectral bands
- **Transfer Learning**: Pre-trained representations for downstream tasks

## Evaluation Framework

### PANGAEA Benchmark

We evaluate Clay using the PANGAEA (A Global and Inclusive Benchmark for Geospatial Foundation Models) framework, which provides standardized evaluation across diverse geospatial tasks.

**PANGAEA Framework Details:**
- **Published**: December 2024 by Marsocci et al.
- **Datasets**: 11 diverse geospatial datasets
- **Tasks**: Semantic segmentation, change detection, regression
- **Geographic Coverage**: Global, addressing geographical bias
- **Code**: Available at https://github.com/VMarsocci/pangaea-bench

### Evaluation Protocol

Our benchmarking follows standard practices for foundation model evaluation:

1. **Frozen Encoder Evaluation**: Clay's pre-trained encoder is frozen, with only a task-specific decoder trained
2. **Standardized Decoder**: UPerNet decoder for dense prediction tasks
3. **Fair Comparison**: Same hyperparameters and training protocols across models
4. **Multiple Runs**: Statistical significance through repeated experiments
5. **Hardware Consistency**: Controlled computational environment

### Datasets Included

The evaluation covers multiple domains and task types:

**Semantic Segmentation:**
- HLS Burn Scars: Wildfire detection using 6-band optical data
- Sen1Floods11: Flood mapping with SAR+optical multimodal data
- MADOS: Marine debris detection
- AI4SmallFarms: Agricultural field detection

**Change Detection:**
- xView2: Building damage assessment
- SpaceNet 7: Building footprint changes

**Regression:**
- BioMassters: Forest biomass estimation using SAR+optical data

## Benchmarking Methodology

### Experimental Design

To ensure fair and transparent benchmarking of Clay Foundation Model, we follow rigorous experimental protocols:

#### 1. **Controlled Environment**
- **Hardware**: NVIDIA RTX 4090 GPU (24GB VRAM), consistent across all experiments
- **Software**: PyTorch Lightning, PANGAEA evaluation framework
- **Randomization**: Fixed random seeds for reproducibility
- **Multiple Runs**: Minimum 5 independent runs for statistical significance

#### 2. **Hyperparameter Optimization**
Following PANGAEA protocols, we will perform systematic hyperparameter search:

```python
# Search space for Clay benchmarking
hyperparameters = {
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    'batch_size': [4, 8, 16],
    'weight_decay': [0.01, 0.05, 0.1],
    'warmup_epochs': [5, 10],
    'total_epochs': [20, 50, 100]
}
```

#### 3. **Statistical Analysis**
- **Metrics**: Mean IoU, pixel accuracy, class-wise IoU with 95% confidence intervals
- **Significance Testing**: Paired t-tests for model comparisons
- **Effect Size**: Cohen's d for practical significance
- **Multiple Comparisons**: Bonferroni correction when comparing multiple models

#### 4. **Fair Comparison Protocol**
- **Frozen Encoder**: Following PANGAEA standard, Clay encoder weights frozen during fine-tuning
- **Identical Decoder**: UPerNet decoder for all models to ensure fair comparison
- **Same Training Data**: Identical train/validation/test splits
- **Band Alignment**: Proper spectral band mapping without zero-padding when possible

### Evaluation Datasets

We will benchmark Clay on the following PANGAEA datasets:

| Dataset | Task | Bands | Resolution | Geographic Coverage |
|---------|------|-------|------------|-------------------|
| HLS Burn Scars | Wildfire Detection | 6 (optical) | 30m | USA |
| Sen1Floods11 | Flood Mapping | 13 optical + 2 SAR | 10m | Global |
| MADOS | Marine Debris | 13 (optical) | 10m | Global |
| PASTIS-R | Crop Segmentation | Multi-temporal | 10m | France |
| BioMassters | Biomass Estimation | SAR + Optical | 10m | Finland |

### Experimental Pipeline

```bash
# 1. Hyperparameter Search (5-fold cross-validation)
python experiments/hyperparameter_search.py \
    --dataset hlsburnscars \
    --model clay \
    --cv_folds 5 \
    --n_trials 50

# 2. Final Evaluation (5 independent runs)
for run in {1..5}; do
    python experiments/benchmark_clay.py \
        --dataset hlsburnscars \
        --model clay \
        --config best_hyperparams.yaml \
        --seed $((42 + run)) \
        --output_dir results/run_${run}
done

# 3. Statistical Analysis
python experiments/statistical_analysis.py \
    --results_dir results/ \
    --output_file clay_benchmark_results.json
```

### Quality Assurance

#### Data Quality
- **Outlier Detection**: Identify and handle anomalous samples
- **Data Leakage**: Ensure no overlap between train/validation/test sets
- **Class Balance**: Report class distributions and handle imbalanced datasets appropriately

#### Model Validation
- **Training Convergence**: Monitor loss curves and early stopping
- **Overfitting Detection**: Compare training vs. validation performance
- **Gradient Flow**: Verify proper gradient propagation during training

#### Reproducibility
- **Version Control**: All code, configs, and results version controlled
- **Environment**: Docker containers for consistent environments
- **Documentation**: Detailed methodology and parameter documentation

### Reporting Standards

#### Performance Metrics
- **Primary Metric**: Mean IoU (intersection over union)
- **Secondary Metrics**: Pixel accuracy, precision, recall, F1-score
- **Statistical Measures**: Mean ± standard deviation across runs
- **Confidence Intervals**: 95% CI for all reported metrics

#### Comparison Framework
- **Baseline Comparison**: Against UNet, ViT, and other PANGAEA models
- **Ablation Studies**: Effect of different components and hyperparameters
- **Error Analysis**: Qualitative analysis of failure cases
- **Computational Cost**: Training time, memory usage, inference speed

### Expected Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Setup | 1 week | Environment setup, data preparation |
| Hyperparameter Search | 2 weeks | Systematic parameter optimization |
| Main Experiments | 3 weeks | Full benchmark evaluation |
| Analysis | 1 week | Statistical analysis, visualization |
| Documentation | 1 week | Results documentation, report writing |

### Success Criteria

The benchmarking will be considered successful if:
1. **Reproducibility**: Results can be reproduced within ±1% mIoU
2. **Statistical Significance**: Clear confidence intervals and significance tests
3. **Fair Comparison**: No systematic bias in experimental design
4. **Comprehensive Coverage**: Results on diverse datasets and tasks
5. **Transparency**: All methodology and code publicly available

This methodology ensures that Clay Foundation Model evaluation will be rigorous, fair, and scientifically sound, providing reliable insights into its capabilities and limitations.

## Results and Comparisons

### Performance on PANGAEA Benchmark

*Note: Results are being compiled through rigorous experimentation following the methodology above. All numbers will be verified through multiple runs and proper statistical analysis.*

**Preliminary Results:**
- [Results to be added after proper experimentation]

### Comparison with Other Foundation Models

Based on published PANGAEA benchmark results from the original paper (Marsocci et al., 2024):

| Model | HLS Burn Scars | Sen1Floods11 | MADOS | Model Type | Notes |
|-------|---------------|--------------|--------|-----------|--------|
| **UNet (baseline)** | **84.51%** | **91.42%** | 54.79% | Supervised CNN | Consistently strong across tasks |
| Prithvi | 83.62% | 90.37% | 49.98% | NASA/IBM MAE | Temporal optical specialty |
| CROMA | 82.42% | 90.89% | **67.55%** | Contrastive MAE | SAR+Optical multimodal |
| DOFA | 80.63% | 89.37% | 59.58% | Multi-modal MAE | Wavelength-adaptive |
| Scale-MAE | 76.68% | 74.13% | 57.32% | Multi-scale MAE | High-resolution focus |
| **Clay** | *TBD* | *TBD* | *TBD* | MAE | **To be benchmarked** |

**Key Findings from PANGAEA Benchmark:**
- **UNet baseline often outperforms foundation models** on simpler tasks
- **Foundation models show advantages on complex multi-modal tasks** (e.g., CROMA on MADOS)
- **Pre-training data characteristics significantly impact performance**
- **No single model dominates across all tasks**

*Clay's performance will be added after proper experimental evaluation following PANGAEA protocols.*

### Key Findings from PANGAEA Literature

According to the original PANGAEA benchmark paper:

1. **Foundation models do not consistently outperform supervised baselines** across all tasks
2. **Pre-training dataset characteristics significantly impact downstream performance**
3. **Spectral richness and spatial resolution of pre-training data matter** for task-specific performance
4. **Limited labeled data scenarios** show some advantages for foundation models
5. **Multimodal datasets remain challenging** for most existing foundation models

## Limitations and Considerations

### Clay-Specific Limitations

- **Single timestamp processing**: Limited temporal modeling capabilities
- **Computational requirements**: Significant GPU memory for large images
- **Domain specificity**: Performance may vary across different geographic regions
- **Band adaptation**: Performance depends on spectral band alignment with pre-training

### General Foundation Model Limitations

- **Not universally superior**: Task-specific models often competitive
- **Geographic bias**: Performance may vary across different regions
- **Class imbalance sensitivity**: Struggles with highly imbalanced datasets
- **Evaluation challenges**: Need for standardized, fair comparison protocols

## Reproducibility

### Code and Data Availability

- **Clay Model**: Available at [repository URL]
- **PANGAEA Framework**: https://github.com/VMarsocci/pangaea-bench
- **Evaluation Scripts**: [To be provided]
- **Pre-trained Weights**: [HuggingFace or similar link]

### Experimental Reproducibility

All experiments can be reproduced using:
```bash
# Example command structure (to be verified)
python -m pangaea.run \
    --config configs/clay_evaluation.yaml \
    --dataset hlsburnscars \
    --model clay \
    --num_runs 5 \
    --seed 42
```

## Future Work

### Planned Improvements

1. **Enhanced temporal modeling**: Integration of time-series capabilities
2. **Improved multimodal fusion**: Better SAR+optical integration
3. **Geographic bias reduction**: Training on more diverse global data
4. **Efficiency optimization**: Reduced computational requirements

### Evaluation Expansion

1. **Additional benchmarks**: Evaluation on other standard benchmarks
2. **Real-world deployment**: Performance in operational settings
3. **Computational efficiency**: Detailed analysis of training and inference costs
4. **Cross-dataset generalization**: Transfer learning capabilities

## References

1. Marsocci, V., et al. "PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models." arXiv preprint arXiv:2412.04204 (2024).

2. [Additional references to be added as experiments are completed]

---

*This document will be updated as experimental results are obtained through rigorous evaluation following the described methodology.*
