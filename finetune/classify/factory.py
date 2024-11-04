import re

import torch
from torch import nn

from src.model import Encoder


class Classifier(nn.Module):
    """
    Classifier class uses Clay Encoder for feature extraction and a head for
    classification.

    Attributes:
        clay_encoder (Encoder): The encoder for feature extraction.
        head (nn.Sequential): The head for classification.
        device (torch.device): The device to run the model on.
    """

    def __init__(self, num_classes=10, ckpt_path=None):
        """
        Initialize the Classifier.

        Args:
            num_classes (int, optional): The number of classes for
            classification. Defaults to 10.
            ckpt_path (str, optional): Clay MAE pretrained model checkpoint
            path. Defaults to None.
        """
        super().__init__()

        # Initialize Clay Encoder with parameters from base model. Set
        # mask_ratio to 0.0 & shuffle to False for downstream tasks.
        # self.clay_encoder = Encoder(
        #     mask_ratio=0.0,
        #     patch_size=8,
        #     shuffle=False,
        #     dim=768,
        #     depth=12,
        #     heads=12,
        #     dim_head=64,
        #     mlp_ratio=4.0,
        # )
        self.clay_encoder = Encoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
            # feature_maps=feature_maps,
            # ckpt_path=ckpt_path,
        )

        # Simple 2 layer MLP head for classification
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

        # Determine the device to run the model on
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Load Clay MAE pretrained weights for the Encoder
        self.load_clay_weights(ckpt_path)

    def load_clay_weights(self, ckpt_path):
        """
        Load the weights for Clay MAE Encoder from a checkpoint file.

        Args:
            ckpt_path (str): Clay MAE pretrained model checkpoint path.
        """
        # Load the checkpoint file
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict")

        # Remove model.encoder prefix for the clay encoder
        state_dict = {
            re.sub(r"^model\.encoder\.", "", name): param
            for name, param in state_dict.items()
            if name.startswith("model.encoder")
        }

        # Copy the weights from the state dict to the encoder
        for name, param in self.clay_encoder.named_parameters():
            if name in state_dict and param.size() == state_dict[name].size():
                param.data.copy_(state_dict[name])  # Copy the weights
            else:
                print(f"No matching parameter for {name} with size {param.size()}")

        # Freeze clay encoder
        for param in self.clay_encoder.parameters():
            param.requires_grad = False

        # Set the encoder to evaluation mode
        self.clay_encoder.eval()

    def forward(self, datacube):
        """
        Forward pass of the Classifier.

        Args:
            datacube (torch.Tensor): A dictionary containing the input datacube
            and meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The output logits.
        """
        # Get the embeddings from the encoder
        embeddings, *_ = self.clay_encoder(
            datacube
        )  # embeddings: batch x (1 + row x col) x 768

        # Use only the first embedding i.e cls token
        embeddings = embeddings[:, 0, :]

        # Pass the embeddings through the head to get the logits
        logits = self.head(embeddings)

        return logits
