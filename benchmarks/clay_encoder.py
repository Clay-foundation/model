import sys
import torch
import torch.nn as nn
from logging import Logger
from pathlib import Path
from typing import Dict, List, Union

sys.path.append('/home/brunosan/code/model')

from src.model import Encoder as ClayEncoder
from src.utils import posemb_sincos_2d_with_gsd
from pangaea.encoders.base import Encoder


class Clay_Encoder(Encoder):
    """Clay Encoder for PANGAEA Benchmark."""

    def __init__(
        self,
        model_name: str = "clay",
        input_bands: Dict[str, List[str]] = None,
        input_size: int = 224,
        embed_dim: int = 768,
        output_layers: List[int] = [3, 5, 7, 11],
        output_dim: Union[int, List[int]] = 768,
        multi_temporal: bool = False,
        multi_temporal_output: bool = False,
        pyramid_output: bool = True,
        encoder_weights: Union[str, Path] = "./pretrained_models/clay_v1.5.0_epoch-07_val-loss-0.1718.ckpt",
        download_url: str = "https://huggingface.co/made-with-clay/Clay/resolve/main/clay_v1.5.0_epoch-07_val-loss-0.1718.ckpt",
        patch_size: int = 8,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: int = 4,
        mask_ratio: float = 0.75,
        in_chans: int = 6,
        **kwargs,
    ):
        if input_bands is None:
            input_bands = {
                "optical": ["B2", "B3", "B4", "B8A", "B11", "B12"]
            }
        
        super().__init__(
            model_name=model_name,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=multi_temporal,
            multi_temporal_output=multi_temporal_output,
            pyramid_output=pyramid_output,
            encoder_weights=encoder_weights,
            download_url=download_url,
        )
        
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.mask_ratio = mask_ratio
        self.in_chans = in_chans
        
        # Create Clay encoder
        self.clay_encoder = ClayEncoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            shuffle=True,
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            dim_head=embed_dim // num_heads,
            mlp_ratio=mlp_ratio,
        )
        
        # Load weights if available
        if self.encoder_weights and Path(self.encoder_weights).exists():
            self.load_encoder_weights(None)

    def load_encoder_weights(self, logger: Logger) -> None:
        """Load Clay encoder weights."""
        try:
            checkpoint = torch.load(self.encoder_weights, map_location="cpu")
            
            # Extract encoder state dict from checkpoint
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                encoder_state_dict = {}
                
                # Filter encoder parameters
                for key, value in state_dict.items():
                    if key.startswith("encoder."):
                        # Remove 'encoder.' prefix
                        new_key = key[8:]
                        encoder_state_dict[new_key] = value
                
                # Load the filtered state dict
                missing_keys, unexpected_keys = self.clay_encoder.load_state_dict(
                    encoder_state_dict, strict=False
                )
                
                if logger:
                    if missing_keys:
                        logger.warning(f"Missing keys when loading Clay encoder: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys when loading Clay encoder: {unexpected_keys}")
                    logger.info("Clay encoder weights loaded successfully")
            else:
                if logger:
                    logger.error("No 'state_dict' found in checkpoint")
                    
        except Exception as e:
            if logger:
                logger.error(f"Error loading Clay encoder weights: {e}")
            else:
                print(f"Error loading Clay encoder weights: {e}")

    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through Clay encoder."""
        # Assume optical data is provided
        optical_data = x["optical"]  # Shape: [B, C, H, W]
        
        B, C, H, W = optical_data.shape
        
        # Handle band selection for Clay encoder (Clay expects 6 specific bands)
        # For datasets with different bands, we need to map or select appropriate bands
        if C != 6:
            # If we have more than 6 bands, select the first 6 for now
            # In a full implementation, you'd want proper band mapping
            if C > 6:
                optical_data = optical_data[:, :6, :, :]  # Take first 6 bands
            else:
                # If we have fewer than 6 bands, pad with zeros
                padding = torch.zeros(B, 6 - C, H, W, device=optical_data.device)
                optical_data = torch.cat([optical_data, padding], dim=1)
            C = 6
        
        # Clay expects specific metadata, using dummy values for now
        device = optical_data.device
        
        # Create dummy metadata (these would normally be provided by the dataset)
        time = torch.zeros(B, 4, device=device)  # Dummy time encoding
        latlon = torch.zeros(B, 4, device=device)  # Dummy lat/lon encoding
        gsd = torch.tensor([10.0], device=device)  # Dummy GSD (10m) - scalar tensor
        
        # Create dummy wavelength information
        # Clay uses wavelength info, we'll use representative values for optical bands
        waves = torch.tensor([
            490.0, 560.0, 665.0, 842.0, 1610.0, 2190.0  # Approximate wavelengths for B2,B3,B4,B8A,B11,B12
        ], device=device)  # [6] - Clay expects a 1D tensor
        
        # Convert patches to embeddings
        patches, waves_encoded = self.clay_encoder.to_patch_embed(optical_data, waves)
        
        # Add position and metadata encodings
        patches_with_encoding = self.clay_encoder.add_encodings(patches, time, latlon, gsd)
        
        # For benchmarking, we don't want masking, so we'll use all patches
        # Add CLS token
        cls_tokens = self.clay_encoder.cls_token.expand(B, -1, -1)
        patches_with_cls = torch.cat([cls_tokens, patches_with_encoding], dim=1)
        
        # Pass through transformer
        encoded_patches = self.clay_encoder.transformer(patches_with_cls)
        
        # Remove CLS token for downstream tasks
        encoded_patches = encoded_patches[:, 1:, :]  # [B, L, D]
        
        # Reshape to spatial format for downstream tasks
        L = encoded_patches.shape[1]
        grid_size = int(L ** 0.5)
        assert grid_size * grid_size == L, f"Cannot reshape {L} patches to square grid"
        
        # Reshape to [B, D, H', W'] format expected by downstream tasks
        encoded_patches = encoded_patches.permute(0, 2, 1)  # [B, D, L]
        encoded_patches = encoded_patches.reshape(B, self.embed_dim, grid_size, grid_size)
        
        # Return list of embeddings for different layers (for now, return same embedding)
        # In full implementation, would extract intermediate layer outputs
        return [encoded_patches for _ in self.output_layers]

    def freeze(self) -> None:
        """Freeze Clay encoder parameters."""
        for param in self.clay_encoder.parameters():
            param.requires_grad = False