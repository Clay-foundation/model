

(model_release)=
# Pretrained Model release v0.0.1

This changelog is a summary of the changes to the pretrained model weights for the Clay model. We follow the "Stanford [Foundation Model Transparency Index](https://github.com/stanford-crfm/fmti)"

Model weights released on 2024/01/12.

> For release notes for the source code, see [](software_release)

### Summary

Clay v0 is a self-supervised modified vision transfer model trained on stacks of Sentinel-2, Sentinel-1 & DEM data. It is trained as a Masked Autoencoder (MAE) to reconstruct the original image from a masked image.

Each data entry is a stack of 13 bands of Sentinel-2, 2 bands of Sentinel-1 & 1 band of DEM data. The model is trained with 3 timesteps of data for each location, with a total of 1203 MGRS tiles globally distributed, each of size 10km x 10km. The data was collected from the Microsoft Planetary Computer.

The model was trained on AWS on 4 NVIDIA A10G GPUs for 25 epochs (~14h per epoch) in December 2024.

Model weights are available on HuggingFace [here](https://huggingface.co/made-with-clay/Clay/).

We also generated embeddings for all trainning data, which can be found on Source Cooperative [here](https://source.coop/).

## Model Architecture

Clat is a Unet, with a modified ViT encoder down to embeddings, and a decoder to reconstruct the masked parts of the original image. The loss function is the MSE between the original image and the reconstructed image.

For details, check the source code [here](https://github.com/Clay-foundation/model/blob/v0.0.1/src/model_clay.py).

![Architecture](assets/architecture.png)

* Core Framework: [Lightning](https://lightning.ai/) and its dependencies, like PyTorch, etc.

* Input modalities:
    * Fixed spec of 13 bands of Sentinel-2, 2 bands of Sentinel-1 & 1 band of DEM data. See below for details.
* Output modalities:
    * As a masked auto-enconder, fixed spec of 13 bands of Sentinel-2, 2 bands of Sentinel-1 & 1 band of DEM data, to mimic the input as close as possible.
* Model size:
    * Number of parameters: `127M`
    * Model size on disk: `~500MB`.
* Model license:
    * Source code: [Apache 2.0](https://github.com/Clay-foundation/model/blob/v0.0.1/LICENSE)
    * Model weights: [OpenRAIL-M](https://github.com/Clay-foundation/model/blob/v0.0.1/LICENSE-MODEL.md)
        * Prohibited uses: See OpenRAIL-M license section 5.
* Feedback and redress mechanisms:
    * Please open an issue or discussion on the [GitHub repository](https://github.com/Clay-foundation/model) or send an email to `bruno@madewithclay.org`.

## Model Card

For v0 of CLAY, we used the [`clay_small`](https://github.com/Clay-foundation/model/blob/0145e55bcf6bd3e9b19f5c07819a1398b6a22c35/src/model_clay.py#L713) setup model.

```
MASKED PATCHES = 75%
INPUT SIZE = 13 bands x 512 width x 512 height
PATCH SIZE = 32 x 32

OPTIMIZER
    Adam
    Learning rate = 1e-4
    Weight decay = 0.05
    Beta 1 = 0.9
    Beta 2 = 0.95

SCHEDULER
    CosineAnnealingWarmRestarts
    T_0 = 1000
    T_mult = 2
    eta_min = Learning rate * 10

ENCODER
    dim = 768
    depth = 12
    heads = 12
    dim_head = 64
    mlp_ratio = 4
    dropout = 0.0
    emb_dropout = 0.0

DECODER
    decoder_dim = 512
    decoder_depth = 8
    decoder_heads = 8
    decoder_dim_head = 64
    decoder_mlp_ratio = 4
    decoder_dropout = 0.0
```

(Data_card)=
## Data Card

We organize our input dataset creation in MGRS tiles. Each tile is a 10km x 10km area. We have `1203` tiles in total, each with 3 timesteps of data between `2017` and `2023`, so `3609 Tiles` in total. Each timestep is a stack of `13` bands of Sentinel-2, 2 bands of Sentinel-1 & 1 band of DEM data. Each tile is split into `512 x 512` chips, so we have around `~1.2 Million` chips in total. Each chip contains `13 bands`, 10 of which are the Sentinel-2 bands, 2 are Sentinel 1 bands & 1 DEM band. We store each chip as geotiff, along with their coordinate & timestamp information that is used for model training.

![Tile location](assets/tiles.png)


* Training dataset size: `6.4 TB`
* Training dataset source links:
    * [Sentinel-2](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)
    * [Sentinel-1](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc)
    * DEM from [Copernicus Digital Elevation Model](https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-90)
* Training dataset items:
    * The actual list of files used is available [here](https://gist.github.com/brunosan/62247e5dc79684bdaca11cefae679e90).
* Data source selection and curation process:
    * We aim for fully open data, with global and historical coverage, with the highest spatial, temporal and spectral resolution, hosted on a cloud format that eases the process to search and download the needed sections.
    * Once these sources are selected, we make a [statistical sample based on cover type](https://github.com/Clay-foundation/model/blob/0145e55bcf6bd3e9b19f5c07819a1398b6a22c35/scripts/landcover.py#L156), so that we have a good coverage of the different landscapes. The land cover data is from [ESA WorldCover 2021](https://registry.opendata.aws/esa-worldcover-vito/).
* Data augmentation:
    * We do not use any data augmentation techniques like affine transformations, random crops (except the masked autoencoder task), etc. We also do not use input mixing like CutMix, MixUp, etc.
    * Clouds, cloud shadows, smog, atmospheric scattering, mid-air planes and other non-ground registrations could be considered natural augmentations. We explicitly filter out large % of clouds on our chips, but small clouds and their shadows might be present. As we increase the number of observations per location, and bands, we expect the model to learn to ignore single events but register patterns (places that are often cloudy or with smog).
* PII or harmful content:
    * We believe that satellites images at this resolution (`10m/px`) are not subject to PII or harmful content concerns.
* Human evaluation, wages, and annotation process:
    * Besides tweaking the statistical samples as part of the model development team, and the stated dataset hosting partners, we do not use any human evaluation, or annotation process, or third party services.

We store each chip as geotiff, along with their coordinate & timestamp information that is used for model training.

![Chips](assets/bands.png)


## Training Card

* Compute Resources:
    * AWS EC2 `g5.12xlarge` with 4 NVIDIA A10G GPUs
* Batch Size:
    * Batch Size = `10`
    * Effective Batch Size = Batch Size x Number of GPUs x Gradient Accumulation Steps = `10` x `4` x `5` = `200`
* Training Time:
    * `25` epochs, each taking ~`15h` to train.
* Carbon Emissions:
    * *Report not yet available from provider, expected March'24*
* Training stages:
    * While developing the model we run small tests locally and on the cloud. We estimate that all testing and development compute is less than the compute used for 1 epoch of training.
    * QA of the model is also done locally and on the cloud, and we estimate that it is less than the compute used for 1 epoch of training.
* Release and distribution:
    * Model development happens in an open source repository on GitHub [here](https://github.com/Clay-foundation/model/).
    * We release the model weights on HuggingFace [here](https://huggingface.co/made-with-clay/Clay/).
    * We release the embeddings on Source Cooperative [here](https://beta.source.coop/clay/).
    * We do not have other distribution channels at this time.
* Production use:
    * We support our partners to build applications with the model, and we expect them to use the model in production.
    * We are developing a web application and expect to release it in 2024 Q1.


![Learning Rate & Epoch](assets/lr.png)

![MSE Loss for Pixel Reconstruction](assets/loss.png)

## Results

As a foundational model, it is designed to be used as a building block for other models. In this section we only a sample of the training objective, which is to reconstruct the original image from a 75% masked image.

[![Reconstruction](assets/reconstruction.png)](assets/reconstruction.png)


### Performance Metrics
The model shows the following performance characteristics for its Masked Autoencoder objective:
* Training loss: `0.52`
* Validation loss: `0.46`

## Known Limitations and Biases

- The model is trained on Sentinel data only.
- Sentinel data only covers land and coastal waters.
- We only train on a ver small sample of the Sentinel archives, both in terms of spatial coverage and time.
- We do not train on the poles, and we do not train on open ocean, nor ocean nor atmospheric volumetric data.
- We do not train on night time data.
- We do not explicitly include extreme events in the training data.
- We only train at most 3 different times per location.


## Ethical Considerations

Our goal is to lower the barrier to use EO data for biodiversity and climate change mitigation and adaptation. We have designed our model to support this goal.

We have also designed our model to be as open as possible, as modular as possible, as undifferentiated and general as possible, and as well documented as possible, so we can maximize the leverage of the resources needed for the creation of this model.

As a fully open model, we cannot however control how it is used. We are aware that EO data can be used for harmful purposes, and we are committed to work with our partners to prevent this from happening.