# Finetuning

Fine-tuning refers to a process in machine learning where a pre-trained model
is further trained on a specific dataset to adapt its parameters to a
downstream task characterized by a relevent domain. It's distinct from training
a model from scratch using the downstream task dataset exclusively.

Related to finetuning in the field of training foundation models is linear
probing, which refers to a technique used to analyze or explore the
representations learned by a pre-text model as it trains. When a large-scale
model (like a vision transformer model) is pre-trained on a vast corpus of
data, it learns rich and complex representations of patterns within the data.
Linear probing involves examining or probing these learned representations by
periodically (e.g. every few epochs of the pre-text model's training cycle)
finetuning a small downstream task on top of the pre-trained model's layers or
embeddings.

We use full finetuning and linear probing in Clay to evaluate the usefulness of
the pre-text model both during its pre-training and afterwards.

Let's take a look at how we are finetuning on the datacube-adapted Cloud to
Street - Microsoft flood benchmark dataset. As a reminder, that is a downstream
segmentation task for identifiyting water pixels in recorded flood events. It's
a binary segmentation problem, specifically.

We process the datacubes into batches formatted in the way the pretrained Clay
model expects, with the addition of information for label images as well.
Here's an example subset of a batch dictionary:

```
{'labels': tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           ...,
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.]]]),
 'pixels': tensor([[[[-0.5994, -0.6108, -0.6034,  ..., -0.5610, -0.5590, -0.5614],
           [-0.5767, -0.5950, -0.6004,  ..., -0.5619, -0.5536, -0.5610],
           [-0.5841, -0.5762, -0.5930,  ..., -0.5491, -0.5304, -0.5373],
           ...,
           [-0.5087, -0.5447, -0.4351,  ..., -0.6162, -0.6083, -0.6044],
           [-0.4184, -0.5432, -0.5003,  ..., -0.6108, -0.6128, -0.6073],
           [-0.2496, -0.5348, -0.5225,  ..., -0.6137, -0.6167, -0.6128]],

          [[-0.6371, -0.6435, -0.6425,  ..., -0.5834, -0.5898, -0.5923],
           [-0.6296, -0.6410, -0.6385,  ..., -0.5794, -0.5983, -0.5958],
           [-0.6167, -0.6177, -0.6182,  ..., -0.5545, -0.5913, -0.5834],
           ...,
           [-0.4800, -0.5153, -0.4308,  ..., -0.6525, -0.6410, -0.6331],
           [-0.4104, -0.5034, -0.4318,  ..., -0.6331, -0.6226, -0.6087],
           [-0.2404, -0.5222, -0.4522,  ..., -0.6231, -0.6241, -0.6177]],

          [[-0.7068, -0.7217, -0.7101,  ..., -0.6118, -0.6178, -0.6290],
           [-0.7087, -0.7022, -0.6924,  ..., -0.6141, -0.6146, -0.6234],
           [-0.7017, -0.6998, -0.6831,  ..., -0.5927, -0.6085, -0.6104],
           ...,
           [-0.5563, -0.5480, -0.4571,  ..., -0.7106, -0.7045, -0.6933],
           [-0.4725, -0.5526, -0.4781,  ..., -0.6975, -0.6789, -0.6807],
           [-0.3117, -0.4995, -0.5000,  ..., -0.6952, -0.6835, -0.6845]],

          ...,
          ]),
 'bbox': tensor([[ 661415., 5369305.,  666535., 5374425.]], dtype=torch.float64),
 'epsg': tensor([32633], dtype=torch.int32),
 'date': ['2020-10-20'],
 'latlon': tensor([[-0.8192, -0.7854]]),
 'timestep': tensor([[-1.2217,  2.7132, -2.4086]]),
 'source_url': ['S2A_L2A_20201022T100051_N0209_R122_T33UXP_20201022T111023_06144-02560_S1B_IW_GRDH_1SDV_20201020T164222_20201020T164247_023899_02D6C4_rtc']}
```

Batches of dictionaries like this run through the Clay model's encoder to
generate embeddings, such as this:

![embedding_ex](https://github.com/Clay-foundation/model/assets/23487320/375c9e83-d539-4730-b923-3b0b61ea689c)

from batches with image bands such as:

![band_red_ex](https://github.com/Clay-foundation/model/assets/23487320/0c254dbf-9589-4fbf-ab32-e3774fbd2f1a)

and labels:

![labels_ex](https://github.com/Clay-foundation/model/assets/23487320/a92eb8e7-9268-46e5-a254-132205cbc498)

These embeddings are reshaped from shape
`batch size * (band groups length * number of patches) * embedding size` to
`batch size * (band groups length * embedding size) patch height * patch width`
before being passed to a series of 2D convolutional transpose and ReLU layers
in a downstream decoder network.

That decoder network is the core of the downstream task. In a forward pass, it
ingests the embeddings, runs them through those layers and computes loss with
respect to the labels. The loss is back-propagated and the decoder gradually
finetunes to the downstream dataset. Here's a peak at the decoder layers:

```
Model(
  (decoder): Sequential(
    (0): Conv2d(4608, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): Upsample(scale_factor=2.0, mode='nearest')
    (2): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): Upsample(scale_factor=2.0, mode='nearest')
    (5): ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Upsample(scale_factor=2.0, mode='nearest')
    (8): ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Upsample(scale_factor=2.0, mode='nearest')
    (11): ConvTranspose2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): Upsample(scale_factor=2.0, mode='nearest')
  )
)
```

Note the absence of an encoder. That is important as this is a finetuning
architecture in which the encoder is replaced by the embeddings from the
pre-trained Clay model.

In comparison, the network we are using to train the downstream task from
scratch looks notably different:

```
Model(
  (encoder): Sequential(
    (0): Conv2d(13, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (decoder): Sequential(
    (0): ConvTranspose2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): ConvTranspose2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): Upsample(scale_factor=2.0, mode='nearest')
    (5): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
```
In this architecture, there is a defined encoder since the embeddings aren't
doing the purpose of encoding latent information.

For both the finetuning and "from scratch" architectures, we use a
`binary_cross_entropy_with_logits` loss function as this is a binary
segmentation problem, and on the predictions, we run sigmoid and max functions
to obtain final segmentation results.

The way we measure relative performance between the finetuned and
"from scratch" model variants happens through calculation of evalution metrics
common for segmentation, such as Dice coefficient, Intersection over Union, F1
score, precision and recall.

### Linear probing

For linear probing, we implement the finetuned architecture in a
[PyTorch callback](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html)
that will execute every `n` epochs during the pre-text model's training.
