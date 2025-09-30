# Classification head fine tuning

We have built an example for training a classification head on top of
the class token embeddings from the frozen Clay encoder.

All the code for this example can be found in the
[classify finetuning folder](https://github.com/Clay-foundation/model/blob/main/finetune/classify)
of this repository.

## Classifier

The `Classifier` class is designed for classification tasks, utilizing the Clay Encoder for feature extraction and adding a classification head on top of it.

### Parameters

- `num_classes (int, optional)`: The number of classes for classification. Defaults to 10.
- `ckpt_path (str, optional)`: Path to the Clay MAE pretrained model checkpoint. Defaults to None.

### Example

In this example, we will use the `Classifier` class to classify images from the [EuroSAT MS dataset](https://github.com/phelber/EuroSAT). The implementation includes data preprocessing, data loading, and model training workflow using [PyTorch Lightning](https://lightning.ai/) & [TorchGeo](https://github.com/microsoft/torchgeo).

In this example we freeze the Clay encoder and only train a very simple 2 layer MLP head for classification. The MLP head receives as input the Clay class token embedding, which already contains the essence of the image as seen by Clay. The model for classification can then be kept very simple while still guaranteeing high quality results.

Notice that the EuroSAT dataset comes without date stamps or location information. The Clay model requires encoded versions of a date stamp and a latitude and longitude information. These values can be set to zero if they are not available, which is what we are doing in the datamodule script.

## Dataset citation

If you have used the EuroSAT dataset, please cite the following papers:

[1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

[2] Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.

Dataset URL: [EuroSAT Dataset](https://madm.dfki.de/files/sentinel/EuroSATallBands.zip)

## Setup

Follow the instructions in the [README](../../README.md) to install the required dependencies.

```bash
git clone <repo-url>
cd model
mamba env create --file environment.yml
mamba activate claymodel
```

## Usage

### Downloading the Dataset

TorchGeo will automatically download the EuroSAT dataset when you run the `classify.py` script.

Directory structure:
```
data/ds
└── images
    └── remote_sensing
        └── otherDatasets
            └── sentinel_2
                └── tif
                    ├── AnnualCrop
                    ├── Forest
                    ├── HerbaceousVegetation
                    ├── Highway
                    ├── Industrial
                    ├── Pasture
                    ├── PermanentCrop
                    ├── Residential
                    ├── River
                    └── SeaLake
```


### Training the Classifcation Head

The model can be run via LightningCLI using configurations in `finetune/classify/configs/classify_eurosat.yaml`.

1. Download the Clay model checkpoint from [Huggingface model hub](https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt) and save it in the `checkpoints/` directory.

2. Modify the batch size, learning rate, and other hyperparameters in the configuration file as needed:
    ```yaml
    data:
      batch_size: 128
      num_workers: 8
    model:
      num_classes: 10
      ckpt_path: checkpoints/clay-v1.5.ckpt
      lr: 1e-4
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

4. Train the classification head:

To ensure that the imports work properly, ensure that the root of
the repository is in the python path before running the script.

```bash
python -m finetune.classify.classify fit --config configs/classify_eurosat.yaml
```

## Acknowledgments

This implementation uses the TorchGeo package for dataset handling and the EuroSAT dataset for training and evaluation. Special thanks to the contributors of [TorchGeo](https://github.com/microsoft/torchgeo) and [EuroSAT](https://github.com/phelber/EuroSAT).
