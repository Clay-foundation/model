# Segmentation head fine tuning

We have built an example for training a segmentation head on top of
feature map extracted from the frozen Clay encoder.

All the code for this example can be found in the
[segment finetuning folder](https://github.com/Clay-foundation/model/blob/main/finetune/segment)
of this repository.

## PANGAEA Benchmark Integration

Clay has been comprehensively benchmarked using the [PANGAEA framework](https://github.com/mithunpaul08/pangaea-bench), demonstrating exceptional performance on multimodal geospatial tasks. Clay is the first foundation model with native SAR+Optical processing capabilities.

**Key Results:**
- **Multimodal Excellence**: 78-85% mIoU on SAR+Optical flood detection (Sen1Floods11)
- **Binary Segmentation**: 75% mIoU on wildfire detection (HLS Burn Scars)
- **Input Flexibility**: Handles 4-15 bands across diverse sensor configurations

For a comprehensive benchmarking tutorial, see [Clay-PANGAEA Benchmark Tutorial](../tutorials/clay-pangaea-benchmark.ipynb).

## Segmentor

The `Segmentor` class extracts the final feature map from the frozen Clay encoder. It then upsamples the feature map to the original image size using a series of convolution & pixel shuffle operations.

### Parameters

- `num_classes (int)`: Number of classes to segment.
- `ckpt_path (str)`: Path to the Clay model checkpoint.

### Example

In this example, we will use the `Segmentor` class to segment Land Use Land Cover (LULC) classes for the Chesapeake Bay CVPR dataset. The implementation includes data preprocessing, data loading, and model training workflow using PyTorch Lightning.

## Dataset citation

If you use this dataset, please cite the associated manuscript:

Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition (CVPR 2019).

Dataset URL: [Chesapeake Bay Land Cover Dataset](https://lila.science/datasets/chesapeakelandcover)

## Setup

Follow the instructions in the [README](../../README.md) to install the required dependencies.

```bash
git clone <repo-url>
cd model
mamba env create --file environment.yml
mamba activate claymodel
```

## Usage

### Preparing the Dataset

Download the Chesapeake Bay Land Cover dataset and organize your dataset directory as recommended.

1. Copy `*_lc.tif` and `*_naip-new.tif` files for segmentation downstream tasks using s5cmd:
   ```bash
   # train
   s5cmd --no-sign-request cp --include "*_lc.tif" --include "*_naip-new.tif" "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-train_tiles/*" data/cvpr/files/train/

   # val
   s5cmd --no-sign-request cp --include "*_lc.tif" --include "*_naip-new.tif" "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover/ny_1m_2013_extended-debuffered-val_tiles/*" data/cvpr/files/val/
   ```

2. Create chips of size `256 x 256` to feed them to the model:
    ```bash
    python finetune/segment/preprocess_data.py data/cvpr/files data/cvpr/ny 256
    ```

Directory structure:
```
data/
└── cvpr/
    ├── files/
    │   ├── train/
    │   └── val/
    └── ny/
        ├── train/
        │   ├── chips/
        │   └── labels/
        └── val/
            ├── chips/
            └── labels/
```

### Training the Model

The model can be run via LightningCLI using configurations in `finetune/segment/configs/segment_chesapeake.yaml`.

1. Download the Clay model checkpoint from [Huggingface model hub](https://huggingface.co/made-with-clay/Clay/blob/main/v1.5/clay-v1.5.ckpt) and save it in the `checkpoints/` directory.

2. Modify the batch size, learning rate, and other hyperparameters in the configuration file as needed:
    ```yaml
    data:
      batch_size: 16
      num_workers: 8
    model:
      num_classes: 7
      ckpt_path: checkpoints/clay-v1.5.ckpt
      lr: 1e-5
      wd: 0.05
      b1: 0.9
      b2: 0.95
    ```

3. Update the [WandB logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger) configuration in the configuration file with your WandB details or use [CSV Logger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html#lightning.pytorch.loggers.CSVLogger) if you don't want to log to WandB:
    ```yaml
    logger:
      - class_path: lightning.pytorch.loggers.WandbLogger
        init_args:
          entity: <wandb-entity>
          project: <wandb-project>
          group: <wandb-group>
          log_model: false
    ```

4. Train the model:

To ensure that the imports work properly, ensure that the root of
the repository is in the python path before running the script.

```bash
python -m finetune.segment.segment fit --config configs/segment_chesapeake.yaml
```
