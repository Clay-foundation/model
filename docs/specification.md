# Pretrained Model release v1.0

This changelog is a summary of the changes to the pretrained model weights for the Clay model. We follow the "Stanford [Foundation Model Transparency Index](https://github.com/stanford-crfm/fmti)"

Model weights released on 2024/05/12.

> For release notes for the source code, see [](software_release)

## Summary

## Model Architecture

## Model Card

## Data Card

The data used for this model is described in detail in the [](training-data) section.

## Normalization parameters

## Training Card

## Results

As a foundational model, it is designed to be used as a building block for other models. In this section we only a sample of the training objective, which is to reconstruct the original image from a 75% masked image.


### Performance Metrics
The model shows the following performance characteristics for its Masked Autoencoder objective:
* Training loss:
* Validation loss:

## Known Limitations and Biases

- Training data for this model only covers land and coastal waters.
- We only train on a ver small sample of the source archives, both in terms of spatial coverage and time.
- We do not train on the poles, and we do not train on open ocean, nor ocean nor atmospheric volumetric data.
- We do not train on night time data.
- We do not explicitly include extreme events in the training data.
- We only train at most 4 different times per location.


## Ethical Considerations

Our goal is to lower the barrier to use EO data for biodiversity and climate change mitigation and adaptation. We have designed our model to support this goal.

We have also designed our model to be as open as possible, as modular as possible, as undifferentiated and general as possible, and as well documented as possible, so we can maximize the leverage of the resources needed for the creation of this model.

As a fully open model, we cannot however control how it is used. We are aware that EO data can be used for harmful purposes, and we are committed to work with our partners to prevent this from happening.
