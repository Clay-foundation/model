# Pretrained Model release v1.5

This changelog is a summary of the changes to the pretrained model weights for the Clay model. We follow the "[Stanford Foundation Model Transparency Index](https://github.com/stanford-crfm/fmti)"

Model weights released on 2024/11/19.

> For release notes for the source code, see [](software_release_v1.5)

## Summary

Clay v1.5 is our MAE-based model designed to handle inputs from a variety of satellite sensors, including Sentinel-2, Landsat, Sentinel-1 SAR, LINZ, NAIP and MODIS. It supports inputs of any size and any number of bands.

### **Acknowledgments and Inspirations:**

Clay v1.5 is based on the foundational work of several pioneering models and research papers. We owe a significant debt of gratitude to the following projects, which provided architectural inspiration and implementation guidance:

- **DOFA**: [Code](https://github.com/zhu-xlab/DOFA), [Paper](https://arxiv.org/abs/2403.15356)
- **GFM**: [Code](https://github.com/mmendiet/GFM), [Paper](https://arxiv.org/abs/2302.04476)
- **Prithvi**: [Code](https://github.com/NASA-IMPACT/hls-foundation-os), [Paper](https://arxiv.org/abs/2310.18660)
- **SatMAE**: [Project](https://sustainlab-group.github.io/SatMAE/)
- **ScaleMAE**: [Project](https://ai-climate.berkeley.edu/scale-mae-website/)
- **Spectral-GPT**: [Paper](https://arxiv.org/abs/2311.07113)

### **Components of Clay v1.5:**

1. **Dynamic Embedding Block**: This component generates patches for the chips from the number of bands and their wavelengths, which are then fed into the masked autoencoder (MAE).
2. **Position Encoding**: This component encodes spatial and temporal information by adding positional encoding to the model. This encoding is scaled according to the Ground Sampling Distance (GSD) and is combined with location information (latitude/longitude) and time step (week/hour).
3. **Masked Autoencoder (MAE)**: A VIT-based MAE is used to reconstruct the sensor data for all input bands. This contributes to 95% of the total loss, known as the reconstruction loss.
4. **Teacher**: DINOv2 is used as a teacher to compute the representation loss, which accounts for the remaining 5% of the total loss.

### **Pre-training and Usage:**

The pre-trained model can process stacks of geospatial data from different sensors with various resolutions and bands, and output vector embeddings. During pre-training, the model processes stacks of chips from different sensors along with metadata such as wavelengths, GSD, latitude/longitude, and time step. The task involves capturing spatial, temporal, and spectral information about Earth and representing these relationships in high-dimensional space. Each resulting embedding represents a specific area of Earth at a particular time.

Clay v1.5 was trained on 70 million globally distributed chips of size 156x256, collected according to the land use/land cover (LULC) statistics of the globe. The training was conducted on AWS using 20 g6.48xlarge instances for ~100 epochs in Sep 2024.

You can access the model weights on HuggingFace [here](https://huggingface.co/made-with-clay/Clay/tree/main/v1.5).

## Model Architecture
![Architecture](https://github.com/Clay-foundation/model/assets/8049519/f6a1e92c-3993-4148-98a2-e3805dae4414)

Clay v1.5's architecture includes a dynamic embedding block for generating patches from multi-band inputs, position encoding to integrate spatial and temporal data, a Vision Transformer-based masked autoencoder (MAE) for reconstructing sensor data, and a DINOv2 teacher model to enhance representation learning. This architecture allows the model to process inputs from various satellite sensors of any size and number of bands, capturing complex geospatial information effectively.

For more details, you can view the source code [here](https://github.com/Clay-foundation/model/blob/main/src/model.py).

https://github.com/Clay-foundation/model/blob/main/LICENSE

- Core Framework: [Lightning](https://lightning.ai/) and its dependencies, such as PyTorch, etc.
- Input modalities:
    - A fixed specification of 10 bands from Sentinel-2, 6 bands from Landsat, 4 bands from NAIP, 3 bands from LINZ, 2 bands from Sentinel-1 data and 7 bands from MODIS.
- Output modalities:
    - As a masked auto-encoder, it has a fixed specification of 10 bands from Sentinel-2, 6 bands from Landsat, 4 bands from NAIP, 3 bands from LINZ, 2 bands from Sentinel-1 data and 7 bands from MODIS to closely mimic the input.
- Model size (in millions):
    - Number of parameters: `632M`
    - Encoder size: `311M`
    - Decoder size: `15M`
    - Teacher size: `304M`
    - Model size on disk (just the encoder): `1.25 GB`.
- Model license:
    - Source code and Model weights: [Apache-2.0](https://github.com/Clay-foundation/model/blob/main/LICENSE)
- Feedback and redress mechanisms:
    - Please open an issue or discussion on the [GitHub repository](https://github.com/Clay-foundation/model) or send an email to `bruno@madewithclay.org`.

## Model Card

For Clay v1.5, we utilized the [`clay_mae_large`](https://github.com/Clay-foundation/model/blob/80012459793cf71f4482b6d0de5254da83f642c6/src/model.py#L608-L624) and the model weights can be found on Huggingface [here](https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt).

```
MASKED PATCHES = 75%
INPUT SIZE = 256

NORM_PIX_LOSS = FALSE
PATCH SIZE = 8

OPTIMIZER
    AdamW
    Learning rate = 1e-5
    Weight decay = 0.05
    Beta 1 = 0.9
    Beta 2 = 0.95

SCHEDULER
    CosineAnnealingWarmRestarts
    T_0 = 1000
    T_mult = 2
    eta_min = Learning rate * 100

ENCODER
    dim = 1024
    depth = 24
    heads = 16
    dim_head = 64
    mlp_ratio = 4

DECODER
    decoder_dim = 512
    decoder_depth = 4
    decoder_heads = 4
    decoder_dim_head = 64
    decoder_mlp_ratio = 4
```

## Data Card

The data used for this model is described in detail in the [](training-data) section.

## Normalization and Wavelength parameters

The normalization parameters depend on the input system that is used. They are
therefore not static values, but rather an input variable to the model.

Similarly, the model takes the central wavelength of each input band as a variable.

During training we used Sentinel-2, Sentinel-1, Landsat 8 and 9, NAIP, LINZ and MODIS data. For these we compiled normalization and wavelength values that were used
during training. These can be used for inferencing when passing data from any of
these systems.

The normalization and wavelength parameters can be found in the following
[metadata file](https://github.com/Clay-foundation/model/blob/main/configs/metadata.yaml).

## Training Card

* Compute Resources:
    * 20 AWS EC2 g6.48xlarge with 8 NVIDIA L4 GPUs each
* Training Time:
    * `100` epochs, each taking ~`8h` to train.
* Carbon Emissions:
    * According to the "Customer Carbon Emission Tool", there were no Scope 1 or Scope 2 carbon emissions. Following the [documentation](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/ccft-estimation.html), we believe this is due to the usage of renewable energy sources. We are aware that Scope 3 emissions might be significant for data centers and that these are not included in the estimate.
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


## Results

As a foundation model, it is designed to be used as a building block for other models. We have documented examples of how the [embedding space](../tutorials/embeddings.ipynb) and the [reconstructions](../tutorials/reconstruction.ipynb) look like for the base model.


### Performance Metrics
The model shows the following performance characteristics for its Masked Autoencoder objective:
* Training loss: `0.165`
* Validation loss: `0.165`

## Known Limitations and Biases

- Training data for this model only covers land and coastal waters.
- We only train on a very small sample of the source archives, both in terms of spatial coverage and time.
- We do not train on the poles, and we do not train on open ocean, nor ocean nor atmospheric volumetric data.
- We do not train on night time data.
- We do not explicitly include extreme events in the training data.
- We only train at most 6 different times per location.


## Ethical Considerations

Our goal is to lower the barrier to use EO data for biodiversity and climate change mitigation and adaptation. We have designed our model to support this goal.

We have also designed our model to be as open as possible, as modular as possible, as undifferentiated and general as possible, and as well documented as possible, so we can maximize the leverage of the resources needed for the creation of this model.

As a fully open model, we cannot however control how it is used. We are aware that EO data can be used for harmful purposes, and we are committed to work with our partners to prevent this from happening.
