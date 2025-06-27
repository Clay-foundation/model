# Clay Foundation Model

## An open source AI model for Earth

Clay is a [foundation model](https://www.adalovelaceinstitute.org/resource/foundation-models-explainer/) of Earth. Foundation models trained on earth observation (EO) data can efficiently distill and synthesize vast amounts of environmental data, allowing them to generalize this knowledge to specific, downstream applications. This makes them versatile and powerful tools for nature and climate applications.

Clay’s model takes satellite imagery, along with information about location and time, as an input, and outputs embeddings, which are mathematical representations of a given area at a certain time on Earth’s surface. It uses a Vision Transformer architecture adapted to understand geospatial and temporal relations on Earth Observation data. The model is trained via Self-supervised learning (SSL) using a Masked Autoencoder (MAE) method.

| Zero to Hero technical Guide (For developers) | Non-Technical User Guide (upcoming webapps) |
|:--:|:--:|
| [![](https://github.com/Clay-foundation/model/assets/434029/0cbfb109-c391-4f73-99f0-abc8769a7a14)](https://www.youtube.com/live/Zd3rbBj56P0?si=_ts3aWBcso0KEH7K) | [![](https://github.com/Clay-foundation/model/assets/434029/5cd209ec-3571-4ae7-98a1-1cef9a898f25)](https://youtu.be/gFjbrGaAL6w?si=ampWKQi9bArVoJsy) |

## Installation

Clay Foundation Model can be easily installed via pip:

```bash
pip install git+https://github.com/Clay-foundation/model.git
```

Download the pretrained Clay v1.5 weights:

```bash
wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
```

## Usage

The Clay model can be used in three main ways:
- **Generate semantic embeddings for any location and time.** You can use embeddings for a variety of tasks, including to:
  - _Find features:_ Locate objects or features, such as surface mines, aquaculture, or concentrated animal feeding operations.

- **Fine-tune the model for downstream tasks such as classification, regression, and generative tasks.** Fine-tuning the model takes advantage of its pre-training to more efficiently classify types, predict values, or detect change than from-scratch methods. Embeddings can also be used to do the following, which require fine-tuning:
  - _Classify types or predict values of interest:_ Identify the types or classes of a given feature, such as crop type or land cover, or predict values of variables of interest, such as above ground biomass or agricultural productivity.
  - _Detect changes over time:_ Find areas that have experienced changes such as deforestation, wildfires, destruction from human conflict, flooding, or urban development.
  - This can be done by training a downstream model to take embeddings as input and output predicted classes/values. This could also include fine-tuning model weights to update the embeddings themselves.

- **Use the model as a backbone for other models.**

## Where is what

- Our **website** is [madewithclay.org](https://madewithclay.org).
- The Clay model **code** lives on [Github](https://github.com/Clay-foundation/model).
  License: [Apache-2.0](https://github.com/Clay-foundation/model/blob/main/LICENSE).
  The latest release is [v1.5](https://github.com/Clay-foundation/model/releases/tag/v1.5)
- The Clay model **weights**  on [Hugging Face](https://huggingface.co/made-with-clay/Clay/).
  License: [Apache-2.0](https://github.com/Clay-foundation/model/blob/main/LICENSE).
- The Clay **documentation** [lives on this site](https://clay-foundation.github.io/model/index.html).
  License: [CC-BY](http://creativecommons.org/licenses/by/4.0/).
- We release the **embeddings** of the used training data on [Source Cooperative](https://beta.source.coop/repositories/clay/clay-model-v0-embeddings).
  License: [ODC-BY](https://opendatacommons.org/licenses/by/).

CLAY is a fiscal sponsored project of the 501c3 non-profit
[Radiant Earth](https://www.radiant.earth).

---
### Table of Contents

```{tableofcontents}
```
