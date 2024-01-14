(software_release)=
# Code Model release v0.0.1

This changelog is a summary of the changes to the source code of the Clay model. 
Released on 2024/01/12.

> For release notes for the trained model, see [](model_release)


### 💫 Highlights

* 🎉 **First release of Clay Foundation Model** 🎉
* Implement MAE with support for position, time, latlon & channel embeddings ([#47](https://github.com/Clay-foundation/model/pull/47))

### 🚀 Model

* Generate embeddings from CLAYModule trained with latlon/time encodings ([#96](https://github.com/Clay-foundation/model/pull/96))
* Add multigpu support & UI to test embeddings ([#109](https://github.com/Clay-foundation/model/pull/109))
* Refactor model for multi-device usage and easier disabling of masking ([#95](https://github.com/Clay-foundation/model/pull/95))
* Rename embeddings file to include MGRS code and store GeoTIFF source_url ([#86](https://github.com/Clay-foundation/model/pull/86))
* Callback function to log Masked Autoencoder reconstructions to WandB ([#88](https://github.com/Clay-foundation/model/pull/88))
* Adapt model to load 512x512 images from s3 bucket ([#85](https://github.com/Clay-foundation/model/pull/85))
* Save embeddings with spatiotemporal metadata to GeoParquet ([#73](https://github.com/Clay-foundation/model/pull/73))
* Generate embeddings via prediction loop ([#56](https://github.com/Clay-foundation/model/pull/56))
* Initial Vision Transformer architecture with MAE decoder ([#37](https://github.com/Clay-foundation/model/pull/37))

### 🗃️ Data Pipeline

* Adapted sampling strategy ([#81](https://github.com/Clay-foundation/model/pull/81))
* Allow ClayDataModule to load GeoTIFF files directly from s3 ([#92](https://github.com/Clay-foundation/model/pull/92))
* Let ClayDataModule return same spatiotemporal fields as GeoTIFFDataModule ([#91](https://github.com/Clay-foundation/model/pull/91))
* Improve date handling for data pipeline ([#76](https://github.com/Clay-foundation/model/pull/76))
* Let LightningDataModule return spatiotemporal metadata ([#66](https://github.com/Clay-foundation/model/pull/66))
* check for no data on a tile level in sentinel 1 vv and vh, sentinel 2 and DEM ([#60](https://github.com/Clay-foundation/model/pull/60))
* Batch setup ([#54](https://github.com/Clay-foundation/model/pull/54))
* LightningDataModule to load GeoTIFF files ([#52](https://github.com/Clay-foundation/model/pull/52))
* Ready for batch ([#44](https://github.com/Clay-foundation/model/pull/44))
* Tiler module ([#41](https://github.com/Clay-foundation/model/pull/41))
* Landcover based sampling strategy ([#29](https://github.com/Clay-foundation/model/pull/29))
* Datacube ([#27](https://github.com/Clay-foundation/model/pull/27))

### 📖 Documentation

* Document how the benchmark dataset labels were prepared ([#100](https://github.com/Clay-foundation/model/pull/100))
* Document how to finetune pretrained model on downstream task ([#99](https://github.com/Clay-foundation/model/pull/99))
* Document how to generate vector embeddings ([#98](https://github.com/Clay-foundation/model/pull/98))
* Document how to run the datacube pipeline with a batch job ([#97](https://github.com/Clay-foundation/model/pull/97))
* Initialize Jupyter Book documentation ([#89](https://github.com/Clay-foundation/model/pull/89))
* Setting the model license to OpenRail-M ([#63](https://github.com/Clay-foundation/model/pull/63))
* Create CODE_OF_CONDUCT.md ([#53](https://github.com/Clay-foundation/model/pull/53))

### 🧰 Maintenance

* Bump pytorch from 2.0.0 to 2.1.0, CUDA from 11.8 to 12.0 ([#51](https://github.com/Clay-foundation/model/pull/51))
* Add pre-commit hooks with ruff formatter/linter rules ([#26](https://github.com/Clay-foundation/model/pull/26))
* Setup GitHub Actions Continuous Integration tests ([#25](https://github.com/Clay-foundation/model/pull/25))
* Setup LightningCLI trainer script ([#24](https://github.com/Clay-foundation/model/pull/24))
* Initial conda environment and binder links ([#15](https://github.com/Clay-foundation/model/pull/15))

### 🧑‍🤝‍🧑 Contributors

* [@brunosan](https://github.com/brunosan)
* [@lillythomas](https://github.com/lillythomas)
* [@srmsoumya](https://github.com/srmsoumya)
* [@yellowcap](https://github.com/yellowcap)
* [@weiji14](https://github.com/weiji14)

**Full Changelog**: https://github.com/Clay-foundation/model/compare/v0.0.0...v0.0.1
