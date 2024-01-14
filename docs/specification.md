

(model_release)=
# Pretrained Model release v0.0.1

This changelog is a summary of the changes to the pretrained model weights for the Clay model. 
Released on 2024/01/12.

> For release notes for the source code, see [](software_release)

### Summary

Clay v0 is a self-supervised modified vision transfer model trained on stacks of Sentinel-2, Sentinel-1 & DEM data. It is trained as a Masked Autoencoder (MAE) to reconstruct the original image from a masked image. 

Each data entry is a stack of 13 bands of Sentinel-2, 2 bands of Sentinel-1 & 1 band of DEM data. The model is trained with 3 timesteps of data for each location, with a total of 1203 MGRS tiles globally distributed, each of size 10km x 10km. The data was collected from the Microsoft Planetary Computer.

The model was trained on AWS on 4 NVIDIA A10G GPUs for 25 epochs (~14h per epoch) in December 2024. 

Model weights are available on HuggingFace [here](https://huggingface.co/made-with-clay/Clay/).

We also generated embeddings for all trainning data, which can be found on Source Cooperative [here](https://source.coop/).

## Model Architecture

![Architecture](assets/architecture.png)

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

Training dataset size: `6.4 TB`  
Number of unique MGRS Tiles: `1203`  
We picked 3 timesteps for each tile, so we have `3609 Tiles` in total. Each MGRS tile covers an area of 10km x 10km. The tiles are [statistically sampled based on cover type](https://github.com/Clay-foundation/model/blob/0145e55bcf6bd3e9b19f5c07819a1398b6a22c35/scripts/landcover.py#L156), so that we have a good coverage of the different landscapes.  

![Tile location](assets/tiles.png)


We then create chips of size `512 x 512` from each tile, so we have around `~1.2 Million` chips in total. Each chip contains `13 bands`, 10 of which are the Sentinel-2 bands, 2 are Sentinel 1 bands & 1 DEM band.
We store each chip as geotiff, along with their coordinate & timestamp information that is used for model training.

![Chips](assets/bands.png)


## Training Card

CLAY v0 `small` is trained on 4 NVIDIA A10G GPUs for 25 epochs on ~1.2 Million chips.  
Effective Batch Size = Batch Size x Number of GPUs x Gradient Accumulation Steps = 10 x 4 x 5 = 200

![Learning Rate & Epoch](assets/lr.png)

![MSE Loss for Pixel Reconstruction](assets/loss.png)

## Results

> CLAY v0 with 75% masked images as input to the model.

![Reconstruction](assets/reconstruction.png)