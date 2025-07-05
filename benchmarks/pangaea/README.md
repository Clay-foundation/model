# Clay Foundation Model Integration with PANGAEA Benchmark

This directory contains the integration of the Clay Foundation Model with the PANGAEA benchmark for evaluating geospatial foundation models.

## Setup

1. **Install PANGAEA benchmark dependencies:**
   ```bash
   cd pangaea-bench
   pip install -r requirements.txt --break-system-packages
   pip install --no-build-isolation --no-deps -e . --break-system-packages
   ```

2. **Clay model checkpoint:**
   The Clay checkpoint is automatically copied to `pangaea-bench/pretrained_models/clay_v1.5.0_epoch-07_val-loss-0.1718.ckpt`

## Running Benchmarks with Clay

### Example Commands

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
   use_wandb=False \
   trainer.fast_dev_run=True
```

#### Semantic Segmentation with HLS Burn Scars:
```bash
cd pangaea-bench
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=hlsburnscars \
   encoder=clay \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   use_wandb=False \
   trainer.fast_dev_run=True
```

## Clay Encoder Configuration

The Clay encoder configuration is located at `pangaea-bench/configs/encoder/clay.yaml` and includes:

- **Model**: Clay Foundation Model v1.5.0
- **Input bands**: B2, B3, B4, B8A, B11, B12 (6 optical bands)
- **Patch size**: 8x8
- **Embed dimension**: 768
- **Architecture**: Vision Transformer with 12 layers, 12 heads

## Implementation Details

- **Band handling**: The Clay encoder automatically handles datasets with different numbers of bands by selecting the first 6 bands or padding with zeros if fewer than 6 bands are available.
- **Metadata**: Uses dummy metadata (time, lat/lon, GSD) for benchmarking purposes.
- **Wavelengths**: Uses representative wavelength values for the 6 optical bands.

## Files Added

1. `pangaea-bench/configs/encoder/clay.yaml` - Clay encoder configuration
2. `pangaea-bench/pangaea/encoders/clay_encoder.py` - Clay encoder implementation
3. `pangaea-bench/pangaea/encoders/__init__.py` - Updated to include Clay encoder

## Notes

- The Clay encoder integration is designed for benchmarking purposes with minimal modifications to the original Clay model.
- For production use, proper band mapping and metadata handling should be implemented based on specific dataset requirements.
- The current implementation supports single-temporal data. Multi-temporal support would require additional modifications.
