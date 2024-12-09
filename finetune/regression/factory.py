"""
Clay Regressor for semantic regression tasks using PixelShuffle.

Attribution:
Decoder inspired by PixelShuffle-based upsampling.
"""

import re

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from src.model import Encoder


class RegressionEncoder(Encoder):
    """
    Encoder class for regression tasks.

    Attributes:
        ckpt_path (str): Path to the clay checkpoint file.
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        ckpt_path=None,
    ):
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
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Load model from checkpoint if provided
        self.load_from_ckpt(ckpt_path)

    def load_from_ckpt(self, ckpt_path):
        """
        Load the model's state from a checkpoint file.

        Args:
            ckpt_path (str): The path to the checkpoint file.
        """
        if ckpt_path:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict")

            # Prepare new state dict with the desired subset and naming
            new_state_dict = {
                re.sub(r"^model\.encoder\.", "", name): param
                for name, param in state_dict.items()
                if name.startswith("model.encoder")
            }

            # Load the modified state dict into the model
            model_state_dict = self.state_dict()
            for name, param in new_state_dict.items():
                if (
                    name in model_state_dict
                    and param.size() == model_state_dict[name].size()
                ):
                    model_state_dict[name].copy_(param)
                else:
                    print(f"No matching parameter for {name} with size {param.size()}")

            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False

    def forward(self, datacube):
        """
        Forward pass of the RegressionEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The embeddings from the final layer.
        """
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        B, C, H, W = cube.shape

        # Patchify and create embeddings per patch
        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        # Transformer encoder
        patches = self.transformer(patches)

        # Remove class token
        patches = patches[:, 1:, :]  # [B, L, D]

        return patches


class Regressor(nn.Module):
    """
    Clay Regressor class that combines the Encoder with PixelShuffle for regression.

    Attributes:
        num_classes (int): Number of output classes for regression.
        ckpt_path (str): Path to the checkpoint file.
    """

    def __init__(self, num_classes, ckpt_path):
        super().__init__()
        # Initialize the encoder
        self.encoder = RegressionEncoder(
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

    def forward(self, datacube):
        """
        Forward pass of the Regressor.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The regression output.
        """
        cube = datacube["pixels"]  # [B C H_in W_in]
        B, C, H_in, W_in = cube.shape

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

        return x
