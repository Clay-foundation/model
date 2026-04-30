import torch
from torch import nn

from claymodel.model import Encoder
from claymodel.utils import load_encoder_weights


class Classifier(nn.Module):
    """
    Classifier class uses Clay Encoder for feature extraction and a head for
    classification.

    Attributes:
        clay_encoder (Encoder): The encoder for feature extraction.
        head (nn.Sequential): The head for classification.
        device (torch.device): The device to run the model on.
    """

    def __init__(self, num_classes: int = 10, ckpt_path: str | None = None) -> None:
        """
        Initialize the Classifier.

        Args:
            num_classes (int, optional): The number of classes for
            classification. Defaults to 10.
            ckpt_path (str, optional): Clay MAE pretrained model checkpoint
            path. Defaults to None.
        """
        super().__init__()

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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load Clay MAE pretrained weights for the Encoder
        if ckpt_path:
            load_encoder_weights(self.clay_encoder, ckpt_path, device=str(self.device))

    def forward(self, datacube: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Classifier.

        Args:
            datacube (torch.Tensor): A dictionary containing the input datacube
            and meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The output logits.
        """
        embeddings, *_ = self.clay_encoder(datacube)
        embeddings = embeddings[:, 0, :]
        return self.head(embeddings)
