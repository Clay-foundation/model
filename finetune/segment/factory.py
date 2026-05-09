"""
Clay Segmentor for semantic segmentation tasks.

Attribution:
Decoder from Segformer: Simple and Efficient Design for Semantic Segmentation
with Transformers
Paper URL: https://arxiv.org/abs/2105.15203
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from claymodel.model import Encoder
from claymodel.utils import load_encoder_weights


class SegmentEncoder(Encoder):
    """
    Encoder class for segmentation tasks, incorporating a feature pyramid
    network (FPN).

    Attributes:
        feature_maps (list): Indices of layers to be used for generating
        feature maps.
        ckpt_path (str): Path to the clay checkpoint file.
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio: float,
        patch_size: int,
        shuffle: bool,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__(
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
        )

        # Set device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Load model from checkpoint if provided
        if ckpt_path:
            load_encoder_weights(self, ckpt_path, device=str(self.device))

    def forward(self, datacube: dict[str, torch.Tensor]) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        B = cube.shape[0]

        # Patchify and create embeddings per patch
        patches, _ = self.to_patch_embed(cube, waves)  # [B L D]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        patches = self.transformer(patches)
        return patches[:, 1:, :]  # [B L D]


class Segmentor(nn.Module):
    """
    Clay Segmentor class that combines the Encoder with FPN layers for semantic
    segmentation.

    Attributes:
        num_classes (int): Number of output classes for segmentation.
        feature_maps (list): Indices of layers to be used for generating feature maps.
        ckpt_path (str): Path to the checkpoint file.
    """

    def __init__(self, num_classes: int, ckpt_path: str | None) -> None:
        super().__init__()
        # Default values are for the clay mae base model.
        self.encoder = SegmentEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
            ckpt_path=ckpt_path,
        )

        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Define layers after the encoder
        D = self.encoder.dim  # embedding dimension
        hidden_dim = 512
        C_out = 64
        r = self.encoder.patch_size  # upscale factor (patch_size)

        self.conv1 = nn.Conv2d(D, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv_ps = nn.Conv2d(hidden_dim, C_out * r * r, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=r)
        self.conv_out = nn.Conv2d(C_out, num_classes, kernel_size=3, padding=1)

    def forward(self, datacube: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Segmentor.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """
        cube = datacube["pixels"]  # [B C H_in W_in]
        _, _, H_in, W_in = cube.shape

        # Get embeddings from the encoder
        patches = self.encoder(datacube)  # [B, L, D]

        # Reshape embeddings to [B, D, H', W']
        H_patches = H_in // self.encoder.patch_size
        W_patches = W_in // self.encoder.patch_size
        x = rearrange(patches, "B (H W) D -> B D H W", H=H_patches, W=W_patches)

        # Pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv_ps(x)  # [B, C_out * r^2, H', W']

        # Upsample using PixelShuffle
        x = self.pixel_shuffle(x)  # [B, C_out, H_in, W_in]

        # Final convolution to get desired output channels
        x = self.conv_out(x)  # [B, num_outputs, H_in, W_in]

        return self.conv_out(x)  # [B, num_outputs, H_in, W_in]
