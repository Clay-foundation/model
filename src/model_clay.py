import os
import re
from typing import Literal

import geopandas as gpd
import lightning as L
import numpy as np
import pandas as pd
import pyarrow as pa
import shapely
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn
from vit_pytorch.vit import Transformer

from src.utils import posemb_sincos_1d, posemb_sincos_2d

torch.set_float32_matmul_precision(precision="medium")


# %%
class Patchify(nn.Module):
    """
    Patchify the input cube & create embeddings per patch
    """

    def __init__(self, in_chans, embed_dim, patch_size):
        """
        Define layers of patch stem.

        Parameters
        ----------
        in_chans : int
            Number of input channels
        embed_dim : int
            Embedding dimension
        patch_size : int
            Patch size
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, xb):
        b, c, h, w = xb.shape
        xb = self.proj(xb)
        xb = rearrange(xb, "b d p1 p2 -> b (p1 p2) d")
        return self.norm(xb)


class Encoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        image_size,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        band_groups,
        dropout,
        emb_dropout,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.band_groups = band_groups
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_group_patches = len(band_groups)
        self.num_patches = self.num_spatial_patches * self.num_group_patches

        # Split the embedding dimensions between spatial & band patches equally
        pos_dim = band_dim = dim // 2

        self.latlon_embedding = nn.Linear(2, dim)
        self.time_embedding = nn.Linear(3, dim)
        self.patch_embedding = nn.ModuleDict(
            {
                name: Patchify(len(bands), dim, patch_size)
                for name, bands in self.band_groups.items()
            }
        )

        # Fix the position & band embedding to sine & cosine functions
        self.register_buffer(
            name="pos_encoding",
            tensor=posemb_sincos_2d(
                h=image_size // patch_size, w=image_size // patch_size, dim=pos_dim
            ),  # [L D/2]
            persistent=False,
        )
        self.register_buffer(
            name="band_encoding",
            tensor=posemb_sincos_1d(
                length=self.num_group_patches, dim=band_dim
            ),  # [G D/2]
            persistent=False,
        )

        # Freeze the weights of position & band encoding
        self.pos_encoding = self.pos_encoding.requires_grad_(False)
        self.band_encoding = self.band_encoding.requires_grad_(False)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * mlp_ratio,
            dropout=dropout,
        )

    def to_patch_embed(self, cube):
        """
        Patchify the input cube & create embeddings per patch

        Parameters
        ----------
        cube : torch.Tensor
            A tensor of shape (B, C, H, W) containing the pixels of the
            datacube.

        Returns
        -------
        patches : torch.Tensor
            A tensor of shape (B, G, L, D) containing the embeddings of the
            patches.
        """
        patches = []
        for name, bands in self.band_groups.items():
            cubeslice = cube[:, bands, :, :]  # [B C H W] -> [B C[slice[...]] H W]
            patches.append(self.patch_embedding[name](cubeslice))

        patches = rearrange(patches, "G B L D -> B G L D")  # [B G L D]
        return patches  # [B G L D]

    def add_encodings(self, patches):
        """
        Add position & band encoding to the patches

        Parameters
        ----------
        patches : torch.Tensor
            A tensor of shape (B, G, L, D) containing the embeddings of the
            patches.

        Returns
        -------
        patches : torch.Tensor
            A tensor of shape (B, G, L, D) containing the embeddings of the
            patches + position & band encoding.
        """
        self.B, G, L, D = patches.shape

        # Align position & band embeddings across patches
        pos_encoding = repeat(
            self.pos_encoding, "L D -> 1 repeat L D", repeat=G
        )  # [1 G L D/2]

        band_encoding = repeat(
            self.band_encoding, "G D -> 1 G repeat D", repeat=L
        )  # [1 G L D/2]

        pos_band_encoding = torch.cat(
            (pos_encoding, band_encoding), dim=-1
        )  # [1 G L D]

        # Add position & band encoding to the input feature vector
        patches = patches + pos_band_encoding  # [B G L D] + [1 G L D] - broadcasting
        patches = self.dropout(patches)  # [B G L D]
        return patches  # [B G L D]

    def embed_metadata(self, patches, latlon, time):
        """
        Add timestep & latlon embedding to the patches

        Parameters
        ----------
        patches : torch.Tensor
            A tensor of shape (B, GL, D) containing the embeddings of the
            patches + position & band encoding.
        latlon : torch.Tensor
            A tensor of shape (B, 2) containing the latlon of the datacube.
        time : torch.Tensor
            A tensor of shape (B, 2) containing the timestep of the datacube.

        Returns
        -------
        patches : torch.Tensor
            A tensor of shape (B, GL, D) containing the embeddings of the
            patches + position & band encoding + timestep & latlon embedding.
        """
        latlon_embedding = rearrange(
            self.latlon_embedding(latlon), "B D -> B 1 D"
        )  # [B D] -> [B 1 D]
        time_embedding = rearrange(
            self.time_embedding(time), "B D -> B 1 D"
        )  # [B D] -> [B 1 D]
        patches = torch.cat(
            [patches, latlon_embedding, time_embedding], dim=1
        )  # [B GL D] + [B 1 D] + [B 1 D] -> [B (GL + 2) D]
        return patches  # [B (GL + 2) D]

    def mask_out(self, patches):
        """
        Mask out patches randomly by shuffling the patches & masking out the
        first N patches

        Parameters
        ----------
        patches : torch.Tensor
            A tensor of shape (B, GL, D) containing the embeddings of the
            patches + position & band encoding + timestep & latlon embedding.

        Returns
        -------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, GL:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.
        masked_matrix : torch.Tensor
            A tensor of shape (B, G, L) containing the mask matrix.
        """
        B, GL, D = patches.shape
        assert (
            GL == self.num_patches
        ), f"Expected {self.num_patches} patches, got {GL} patches."

        if self.shuffle:  # Shuffle the patches
            noise = torch.randn((B, GL), device=patches.device)  # [B GL]
        else:  # Don't shuffle useful for interpolation & inspection of embeddings
            noise = rearrange(
                torch.arange(B * GL, device=patches.device), "(B GL) -> B GL", B=B
            )

        random_indices = torch.argsort(noise, dim=-1)  # [B GL]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B GL]

        num_masked_patches = int(
            self.mask_ratio * self.num_patches
        )  # Number of patches to be masked out
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],  # [B mask_ratio * GL]
            random_indices[:, num_masked_patches:],  # [B (1 - mask_ratio) * GL]
        )

        # create a mask of shape B G L, where 1 indicates a masked patch
        # and 0 indicates an unmasked patch
        masked_matrix = torch.zeros((B, GL), device=patches.device)  # [B GL] = 0
        masked_matrix[:, :num_masked_patches] = 1  # [B mask_ratio * GL] = 1
        masked_matrix = torch.gather(
            masked_matrix, dim=1, index=reverse_indices
        )  # [B GL] -> [B GL] - reorder the patches
        masked_matrix = rearrange(
            masked_matrix,
            "B (G L) -> B G L",
            G=self.num_group_patches,  # [B G L]
        )

        # mask out the patches
        batch_indices = rearrange(
            torch.arange(B, device=patches.device), "B -> B 1"
        )  # [B 1]
        unmasked_patches = patches[
            batch_indices, unmasked_indices, :
        ]  # [B GL:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B GL:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B GL:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B G L]

    def forward(self, datacube):
        cube, time, latlon = (
            datacube["pixels"],
            datacube["timestep"],
            datacube["latlon"],
        )  # [B C H W]

        B, C, H, W = cube.shape

        patches = self.to_patch_embed(
            cube
        )  # [B G L D] - patchify & create embeddings per patch

        patches = self.add_encodings(
            patches
        )  # [B G L D] - add position & band encoding to the embeddings

        patches = rearrange(patches, "B G L D -> B (G L) D")  # [B (GL) D]
        patches = self.dropout(patches)  # [B (GL) D]

        # mask out patches
        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.mask_out(
            patches
        )  # [B GL:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B G L]

        # add timestep & latlon embedding to only the unmasked patches
        unmasked_patches = self.embed_metadata(
            unmasked_patches, latlon, time
        )  # [B (GL:(1 - mask_ratio) + 2) D]

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches = self.transformer(
            unmasked_patches
        )  # [B (GL:(1 - mask_ratio) + 2) D]

        return (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B (GL:(1 - mask_ratio) + 2) D], [(1-mask_ratio)], [mask_ratio], [B G L]


class Decoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        image_size,
        patch_size,
        encoder_dim,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        band_groups,
        dropout,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.dim = dim
        self.band_groups = band_groups
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_group_patches = len(band_groups)
        self.num_patches = self.num_spatial_patches * self.num_group_patches

        self.enc_to_dec = (
            nn.Linear(encoder_dim, dim) if encoder_dim != dim else nn.Identity()
        )
        self.mask_patch = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * mlp_ratio,
            dropout=dropout,
        )

        # Split the embedding dimensions between spatial & band patches equally
        pos_dim = band_dim = dim // 2

        # Fix the position & band embedding to sine & cosine functions
        self.register_buffer(
            name="pos_encoding",
            tensor=posemb_sincos_2d(
                h=image_size // patch_size, w=image_size // patch_size, dim=pos_dim
            ),  # [L D/2]
            persistent=False,
        )
        self.register_buffer(
            name="band_encoding",
            tensor=posemb_sincos_1d(
                length=self.num_group_patches, dim=band_dim
            ),  # [G D/2]
            persistent=False,
        )

        # Freeze the weights of position & band encoding
        self.pos_encoding = self.pos_encoding.requires_grad_(False)
        self.band_encoding = self.band_encoding.requires_grad_(False)

        self.embed_to_pixels = nn.ModuleDict(
            {
                name: nn.Linear(dim, (patch_size**2) * len(bands))
                for name, bands in self.band_groups.items()
            }
        )

    def reconstruct_and_add_encoding(
        self, unmasked_patches, unmasked_indices, masked_indices
    ):
        """
        Reconstruct the input patches from the random mask patch & add position
        & band encoding to them.

        Parameters
        ----------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, GL:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.

        Returns
        -------
        decoder_patches : torch.Tensor
            A tensor of shape (B, GL, D) containing the embeddings for the
            decoder part of the model.
        """
        B, *_ = unmasked_patches.shape

        # Align position & band embeddings across patches
        pos_encoding = repeat(
            self.pos_encoding, "L D -> 1 repeat L D", repeat=self.num_group_patches
        )  # [1 G L D/2]
        band_encoding = repeat(
            self.band_encoding, "G D -> 1 G repeat D", repeat=self.num_spatial_patches
        )  # [1 G L D/2]

        pos_band_encoding = torch.cat(
            (pos_encoding, band_encoding), dim=-1
        )  # [1 G L D]
        pos_band_encoding = rearrange(
            pos_band_encoding, "1 G L D -> 1 (G L) D"
        )  # [1 (GL) D]
        pos_band_encoding = repeat(
            pos_band_encoding, "1 (GL) D -> B (GL) D", B=B
        )  # [B (GL) D]

        batch_indices = rearrange(
            torch.arange(B, device=unmasked_patches.device), "B -> B 1"
        )  # [B 1]
        unmasked_pos_band_encoding = pos_band_encoding[
            batch_indices, unmasked_indices, :
        ]  # [B (GL:(1 - mask_ratio)) D]
        masked_pos_band_encoding = pos_band_encoding[
            batch_indices, masked_indices, :
        ]  # [B (GL:mask_ratio) D]

        # Reconstruct the masked patches from the random mask patch &
        # add position & band encoding to them
        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_patches = repeat(
            self.mask_patch, "D -> B GL D", B=B, GL=num_masked_patches
        )  # [B GL:mask_ratio D]
        masked_patches = (
            masked_patches + masked_pos_band_encoding
        )  # [B GL:mask_ratio D] + [B GL:mask_ratio D]

        # Add position & band encoding to the unmasked patches
        unmasked_patches = (
            unmasked_patches + unmasked_pos_band_encoding
        )  # [B GL:(1 - masked_ratio) D] + [B GL:(1 - mask_ratio) D]

        # Concatenate the masked & unmasked patches
        decoder_patches = torch.zeros(
            (B, self.num_patches, self.dim), device=unmasked_patches.device
        )  # [B GL D]
        decoder_patches[batch_indices, unmasked_indices, :] = (
            unmasked_patches  # [B GL:(1 - mask_ratio) D]
        )
        decoder_patches[batch_indices, masked_indices, :] = (
            masked_patches  # [B GL:mask_ratio D]
        )

        return decoder_patches  # [B GL D]

    def pixelify(self, patches):
        """
        Convert the patches into pixel space to compute the loss

        Parameters
        ----------
        patches : torch.Tensor
            A tensor of shape (B, GL, D) containing the embeddings from the
            decoder part of the model.

        Returns
        -------
        pixels : torch.Tensor
            A tensor of shape (B, C, L, PP) containing the pixels of the
            datacube.
        """
        patches = rearrange(
            patches, "B (G L) D -> B G L D", G=len(self.band_groups)
        )  # [B G L D]
        pixels = []
        for i, (name, bands) in enumerate(self.band_groups.items()):
            group_embeddings = patches[:, i, :, :]  # [B L D]
            group_pixels = self.embed_to_pixels[name](group_embeddings)  # [B L (P P C)]
            group_pixels = rearrange(
                group_pixels,
                "B L (PP C) -> B C L PP",
                PP=(self.patch_size**2),
            )  # [B C L PP]
            pixels.append(group_pixels)  # [B C L PP]

        pixels = torch.cat(pixels, dim=1)  # [B C L PP]
        return pixels  # [B C L PP]

    def forward(self, encoded_unmasked_patches, unmasked_indices, masked_indices):
        # Change the embedding dimension from encoder to decoder
        encoded_unmasked_patches = self.enc_to_dec(encoded_unmasked_patches)

        # Split the patches into encoded unmasked patches & meta patches
        encoded_unmasked_patches, encoded_unmasked_meta_patches = (
            encoded_unmasked_patches[:, :-2, :],
            encoded_unmasked_patches[:, -2:, :],
        )  # [B (GL:(1 - mask_ratio)) D], [B 2 D]

        # Reconstruct the patches to feed into the decoder transformer
        decoder_patches = self.reconstruct_and_add_encoding(
            encoded_unmasked_patches, unmasked_indices, masked_indices
        )  # [B GL D]

        # Add the metadata patches back to the decoder patches
        decoder_patches = torch.cat(
            [decoder_patches, encoded_unmasked_meta_patches], dim=1
        )  # [B (GL + 2) D]

        # Pass the decoder patches through the transformer
        decoded_patches = self.transformer(decoder_patches)  # [B (GL + 2) D]

        # Remove the metadata patches from the decoded patches
        decoded_patches = decoded_patches[:, :-2, :]  # [B GL D]

        # Piixelify the decoded patches
        pixels = self.pixelify(decoded_patches)  # [B C L PP]
        return pixels


class CLAY(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        image_size,
        patch_size,
        shuffle,
        # ENCODER
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        dropout,
        emb_dropout,
        # DECODER
        decoder_dim,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        decoder_mlp_ratio,
        decoder_dropout,
        # EO
        band_groups={
            "rgb": (2, 1, 0),
            "rededge": (3, 4, 5, 7),
            "nir": (6,),
            "swir": (8, 9),
            "sar": (10, 11),
            "dem": (12,),
        },
        **kwargs,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.band_groups = band_groups

        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            image_size=image_size,
            patch_size=patch_size,
            shuffle=shuffle,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            band_groups=band_groups,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.decoder = Decoder(
            mask_ratio=mask_ratio,
            image_size=image_size,
            patch_size=patch_size,
            encoder_dim=dim,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_ratio=decoder_mlp_ratio,
            band_groups=band_groups,
            dropout=decoder_dropout,
        )

    def per_pixel_loss(self, cube, pixels, masked_matrix):
        """
        Compute the per pixel loss

        Parameters
        ----------
        cube : torch.Tensor
            A tensor of shape (B, C, H, W) containing the pixels of the
            datacube.
        pixels : torch.Tensor
            A tensor of shape (B, C, L, PP) containing the pixels per patch of
            the datacube.
        masked_matrix : torch.Tensor
            A tensor of shape (B, G, L) containing the mask matrix.

        Returns
        -------
        loss
        """
        patches = rearrange(
            cube,
            "B C (h p1) (w p2) -> B C (h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )  # [B C L PP]

        # loss = (patches - pixels) ** 2  # loss per pixel
        loss = F.mse_loss(patches, pixels, reduction="none")  # loss per pixel
        loss = reduce(loss, "B C L PP -> B C L", reduction="mean")  # loss per patch

        # mask out the loss for unmasked patches
        actual_loss, masked_patches_in_group = 0.0, 0.0
        for i, (name, group) in enumerate(self.band_groups.items()):
            group_loss = reduce(
                loss[:, group, :], "B G L -> B L", "mean"
            )  # (B, L) - loss per group
            actual_loss += (
                group_loss * masked_matrix[:, i]
            ).sum()  # (B, L) * (B, L) -> (B, L) -> (B) -> scalar
            masked_patches_in_group += masked_matrix[
                :, i
            ].sum()  # (B, L) -> (B) -> scalar

        return actual_loss / masked_patches_in_group

    def forward(self, datacube):
        # ENCODER
        (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.encoder(
            datacube
        )  # [B (GL:(1 - mask_ratio) + 2) D], [(1-mask_ratio)], [mask_ratio], [B G L]

        # DECODER
        pixels = self.decoder(
            encoded_unmasked_patches, unmasked_indices, masked_indices
        )  # [B C L PP]

        # LOSS
        loss = self.per_pixel_loss(datacube["pixels"], pixels, masked_matrix)

        return loss


def clay_tiny(**kwargs):
    args = {
        # ENCODER
        "dim": 256,
        "depth": 4,
        "heads": 4,
        "dim_head": 64,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # DECODER
        "decoder_dim": 128,
        "decoder_depth": 2,
        "decoder_heads": 2,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 2,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    model = CLAY(**args)
    return model


def clay_small(**kwargs):
    args = {
        # ENCODER
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # DECODER
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 8,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    model = CLAY(**args)
    return model


def clay_medium(**kwargs):
    args = {
        # ENCODER
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # DECODER
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    model = CLAY(**args)
    return model


def clay_large(**kwargs):
    args = {
        # ENCODER
        "dim": 1280,
        "depth": 32,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        # DECODER
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
        "decoder_dropout": 0.0,
    }
    args.update(kwargs)
    model = CLAY(**args)
    return model


class CLAYModule(L.LightningModule):
    def __init__(  # noqa: PLR0913
        self,
        model_size="small",
        mask_ratio=0.75,
        image_size=512,
        patch_size=32,
        shuffle=False,
        lr=1e-4,
        wd=0.05,
        b1=0.9,
        b2=0.95,
        embeddings_level: Literal["mean", "patch", "group"] = "mean",
        band_groups={
            "rgb": (2, 1, 0),
            "rededge": (3, 4, 5, 7),
            "nir": (6,),
            "swir": (8, 9),
            "sar": (10, 11),
            "dem": (12,),
        },
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        model_map = {
            "tiny": clay_tiny,
            "small": clay_small,
            "medium": clay_medium,
            "large": clay_large,
        }
        if model_size in model_map:
            model_args = {
                "mask_ratio": mask_ratio,
                "image_size": image_size,
                "patch_size": patch_size,
                "shuffle": shuffle,
            }
            if band_groups:
                model_args["band_groups"] = band_groups
            self.model = model_map[model_size](**model_args)
        else:
            raise ValueError(
                f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            )

    def forward(self, cube: dict[str, torch.Tensor]):
        return self.model(cube)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1000, T_mult=2, eta_min=self.hparams.lr * 100, last_epoch=-1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch: dict[str, torch.Tensor], batch_idx: int, phase: str):
        cube = batch
        loss = self(cube)
        self.log(
            name=f"{phase}/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def predict_step(
        self, batch: dict[str, torch.Tensor | list[str]], batch_idx: int
    ) -> gpd.GeoDataFrame:
        """
        Logic for the neural network's prediction loop.
        """
        # Get image, bounding box, EPSG code, and date inputs
        # x: torch.Tensor = batch["pixels"]  # image of shape (1, 13, 512, 512) # BCHW
        bboxes: np.ndarray = batch["bbox"].cpu().__array__()  # bounding boxes
        epsgs: torch.Tensor = batch["epsg"]  # coordinate reference systems as EPSG code
        dates: list[str] = batch["date"]  # dates, e.g. ['2022-12-12', '2022-12-12']
        source_urls: list[str] = batch[  # URLs, e.g. ['s3://1.tif', 's3://2.tif']
            "source_url"
        ]

        # Forward encoder
        self.model.encoder.mask_ratio = 0.0  # disable masking
        outputs_encoder: dict = self.model.encoder(
            datacube=batch  # input (pixels, timestep, latlon)
        )

        # Get embeddings generated from encoder
        # (encoded_unmasked_patches, _, _, _) = outputs_encoder
        embeddings_raw: torch.Tensor = outputs_encoder[0]
        assert embeddings_raw.shape == torch.Size(
            [self.model.encoder.B, 1538, 768]  # (batch_size, seq_length, hidden_size)
        )
        assert not torch.isnan(embeddings_raw).any()  # ensure no NaNs in embedding

        if self.hparams.embeddings_level == "mean":
            # Take the mean of the embeddings along the sequence_length dimension
            # excluding the last two latlon_ and time_ embeddings, i.e. compute
            # mean over patch embeddings only
            embeddings_output: torch.Tensor = embeddings_raw[:, :-2, :].mean(dim=1)
            expected_size = [self.model.encoder.B, 768]  # (batch_size, hidden_size)
        elif self.hparams.embeddings_level in ["patch", "group"]:
            # Take the mean of the embeddings along the group dimension
            # excluding the last two latlon_ and time_ embeddings. This
            # results in one embedding per patch.
            embeddings_output = rearrange(
                embeddings_raw[:, :-2, :], "b (g h w) d -> b g h w d", w=16, h=16, g=6
            )
            if self.hparams.embeddings_level == "patch":
                embeddings_output = reduce(
                    embeddings_output, "b g h w d -> b h w d", "mean"
                )
                expected_size = [
                    self.model.encoder.B,
                    16,
                    16,
                    768,
                ]
            else:
                expected_size = [
                    self.model.encoder.B,
                    6,
                    16,
                    16,
                    768,
                ]
        else:
            raise ValueError(
                f"Value {self.hparams.embeddings_level} no allowed. "
                "Choose one from mean, patch, or group"
            )

        assert embeddings_output.shape == torch.Size(expected_size)

        # Create table to store the embeddings with spatiotemporal metadata
        unique_epsg_codes = set(int(epsg) for epsg in epsgs)
        if len(unique_epsg_codes) == 1:  # check that there's only 1 unique EPSG
            epsg: int = batch["epsg"][0]
        else:
            raise NotImplementedError(
                f"More than 1 EPSG code detected: {unique_epsg_codes}"
            )

        gdf = gpd.GeoDataFrame(
            data={
                "source_url": pd.Series(data=source_urls, dtype="string[pyarrow]"),
                "date": pd.to_datetime(arg=dates, format="%Y-%m-%d").astype(
                    dtype="date32[day][pyarrow]"
                ),
                "embeddings": pa.FixedShapeTensorArray.from_numpy_ndarray(
                    np.ascontiguousarray(embeddings_output.cpu().detach().__array__())
                ),
            },
            geometry=shapely.box(
                xmin=bboxes[:, 0],
                ymin=bboxes[:, 1],
                xmax=bboxes[:, 2],
                ymax=bboxes[:, 3],
            ),
            crs=f"EPSG:{epsg}",
        )
        gdf = gdf.to_crs(crs="OGC:CRS84")  # reproject from UTM to lonlat coordinates

        return gdf

    def on_predict_epoch_end(self) -> gpd.GeoDataFrame:
        """
        Logic to gather all the results from one epoch in a prediction loop.
        """
        # Combine list of geopandas.GeoDataFrame objects
        results: list[gpd.GeoDataFrame] = self.trainer.predict_loop.predictions
        if results:
            gdf: gpd.GeoDataFrame = pd.concat(
                objs=results, axis="index", ignore_index=True
            )
        else:
            print(
                "No embeddings generated, "
                f"possibly no GeoTIFF files in {self.trainer.datamodule.data_dir}"
            )
            return

        # Save embeddings in GeoParquet format, one file for each MGRS code
        outfolder: str = f"{self.trainer.default_root_dir}/data/embeddings"
        os.makedirs(name=outfolder, exist_ok=True)

        # Find unique MGRS names (e.g. '12ABC'), e.g.
        # from 's3://.../.../claytile_12ABC_20201231_v02_0001.tif', get 12ABC
        mgrs_codes = gdf.source_url.str.split("/").str[-1].str.split("_").str[1]
        unique_mgrs_codes = mgrs_codes.unique()
        for mgrs_code in unique_mgrs_codes:
            if re.match(pattern=r"(\d{2}[A-Z]{3})", string=mgrs_code) is None:
                raise ValueError(
                    "MGRS code should have 2 numbers and 3 letters (e.g. 12ABC), "
                    f"but got {mgrs_code} instead"
                )

            # Subset GeoDataFrame to a single MGRS code
            _gdf: gpd.GeoDataFrame = gdf.loc[mgrs_codes == mgrs_code].reset_index()

            # Get min/max date from GeoDataFrame
            minmax_date: pd.Series = _gdf.date.agg(func=["min", "max"])
            min_date: str = minmax_date["min"].strftime("%Y%m%d")
            max_date: str = minmax_date["max"].strftime("%Y%m%d")

            # Output to a GeoParquet filename like
            # {MGRS:5}_{MINDATE:8}_{MAXDATE:8}_v{VERSION:3}.gpq
            outpath = f"{outfolder}/{mgrs_code}_{min_date}_{max_date}_v001.gpq"
            _gdf.to_parquet(path=outpath, compression="ZSTD", schema_version="1.0.0")
            print(
                f"Saved {len(_gdf)} rows of embeddings of "
                f"shape {gdf.embeddings.iloc[0].shape} to {outpath}"
            )

        return gdf
