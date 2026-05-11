"""
Clay Regressor for semantic regression tasks using PixelShuffle.

Attribution:
Decoder inspired by PixelShuffle-based upsampling.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from claymodel.model import Encoder
from claymodel.utils import load_encoder_weights


class Regressor(nn.Module):
    """
    Clay Regressor class that combines the Encoder with PixelShuffle for regression.

    Attributes:
        num_classes (int): Number of output classes for regression.
        ckpt_path (str): Path to the checkpoint file.
    """

    def __init__(self, num_classes: int, ckpt_path: str | None) -> None:
        super().__init__()
        # Initialize the encoder
        self.encoder = Encoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
        )

        # Set device and load pretrained weights
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if ckpt_path:
            load_encoder_weights(self.encoder, ckpt_path, device=str(device))

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
        Forward pass of the Regressor.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The regression output.
        """
        cube = datacube["pixels"]  # [B C H_in W_in]
        _, _, H_in, W_in = cube.shape

        # Get embeddings from the encoder (strip CLS token)
        encoded, *_ = self.encoder(datacube)
        patches = encoded[:, 1:, :]  # [B, L, D]

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
        return self.conv_out(x)  # [B, num_outputs, H_in, W_in]
