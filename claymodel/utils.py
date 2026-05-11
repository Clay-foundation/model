"""
Utilities for Clay Foundation Model.

Position embedding code adapted from:
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
"""

__all__ = [
    "load_encoder_weights",
    "posemb_sincos_1d",
    "posemb_sincos_2d",
    "posemb_sincos_2d_with_gsd",
]

import re

import torch
from torch import nn


def posemb_sincos_2d(
    h: int,
    w: int,
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d_with_gsd(
    h: int,
    w: int,
    dim: int,
    gsd: torch.Tensor | None = None,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    if gsd is None:
        gsd = torch.tensor(1.0)
    gsd = gsd.to(x.device)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)  # Adjusted for g

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def load_encoder_weights(
    encoder: nn.Module,
    ckpt_path: str,
    device: str = "cpu",
    freeze: bool = True,
) -> tuple[list[str], list[str]]:
    """Load Clay encoder weights from a full MAE checkpoint.

    Extracts encoder weights from the full MAE checkpoint (which includes
    encoder, decoder, teacher, and projection layers) and loads them into
    the provided encoder module.

    This is the shared utility used by all finetune factories. It uses
    state_dict()-based loading to correctly handle both parameters and
    registered buffers.

    Args:
        encoder: An Encoder (or nn.Module subclass) to load weights into.
        ckpt_path: Path to the Clay MAE checkpoint file.
        device: Device to load weights onto.
        freeze: If True, freeze all loaded parameters and set eval mode.

    Returns:
        Tuple of (loaded_keys, skipped_keys) for diagnostic purposes.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Extract encoder weights and strip the "model.encoder." prefix
    encoder_state_dict = {
        re.sub(r"^model\.encoder\.", "", name): param
        for name, param in state_dict.items()
        if name.startswith("model.encoder")
    }

    # Load matching weights into the encoder's state dict
    model_state_dict = encoder.state_dict()
    loaded_keys = []
    skipped_keys = []

    for name, param in encoder_state_dict.items():
        if name in model_state_dict and param.size() == model_state_dict[name].size():
            model_state_dict[name].copy_(param)
            loaded_keys.append(name)
        else:
            skipped_keys.append(name)

    if freeze:
        for name, param in encoder.named_parameters():
            if name in encoder_state_dict:
                param.requires_grad = False
        encoder.eval()

    return loaded_keys, skipped_keys


def posemb_sincos_1d(
    waves: torch.Tensor | int,
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert dim % 2 == 0, "Feature dimension must be a multiple of 2 for sincos embedding"
    waves = torch.arange(waves) if isinstance(waves, int) else waves

    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_waves = waves[:, None] * omega[None, :]
    pe = torch.cat((scaled_waves.sin(), scaled_waves.cos()), dim=1)

    return pe.type(dtype)
