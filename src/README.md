# Clay Foundation Model Modules

This folder contains several LightningDataModule, LightningModule and callback
classes.

## DataModules (data pipeline)

- datamodule.py - Data pipeline to read in Earth Observation chips from GeoTIFF files

## LightningModule (model architecture)

- model_clay.py - Clay Foundation Model architecture with spatiotemporal encoders
- model_vit.py - Vanilla Vision Transformer neural network model architecture

## Callbacks (custom plugins)

- callbacks_wandb.py - Log metrics and predictions to Weights and Biases while training.

## References

- https://lightning.ai/docs/pytorch/2.1.0/data/datamodule.html
- https://lightning.ai/docs/pytorch/2.1.0/common/lightning_module.html
- https://lightning.ai/docs/pytorch/2.1.0/extensions/callbacks.html
