import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from torch import nn
from vit_pytorch.simple_vit import Transformer

from src.utils import posemb_sincos_2d_with_gsd
from src.factory import DynamicEmbedding


def encode_metadata(time, latlon, has_time=True, has_lat_lon=True):
    """Encode metadata in batch
    time: [B 2] hour week
    latlon: [B 2] lat lon
    """
    B = time.size(0)
    encoded_values = torch.zeros(B, 8, dtype=torch.float32)

    if has_lat_lon:
        lat, lon = latlon[:, 0], latlon[:, 1]
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180

        encoded_values[:, 0] = torch.sin(lat)
        encoded_values[:, 1] = torch.cos(lat)
        encoded_values[:, 2] = torch.sin(lon)
        encoded_values[:, 3] = torch.cos(lon)
    else:
        encoded_values[:, :4] = 0

    if not has_time:
        hour, week = time[:, 0], time[:, 1]
        hour = hour * 2 * np.pi / 24
        week = week * 2 * np.pi / 52

        encoded_values[:, 4] = torch.sin(hour)
        encoded_values[:, 5] = torch.cos(hour)
        encoded_values[:, 6] = torch.sin(week)
        encoded_values[:, 7] = torch.cos(week)
    else:
        encoded_values[:, 4:] = 0

    return encoded_values


class Encoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
        )

    def to_patch_embed(self, cube, waves):
        """Split the input cube into patches & create embeddings per patch"""
        patches, waves_encoded = self.patch_embedding(cube, waves)  # [B L D]
        return patches, waves_encoded  # ([B L D], [N D])

    def add_encodings(self, patches, time, latlon, gsd):
        """Add position encoding to the patches"""
        B, L, D = patches.shape

        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size,
                w=grid_size,
                dim=(self.dim - 8),
                gsd=gsd[0],
            )
            .to(patches.device)
            .detach()
        )  # [L (D - 8)]

        metadata = (
            encode_metadata(time, latlon, has_time=True, has_lat_lon=True)
            .to(patches.device)
            .detach()
        )  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        metadata = repeat(metadata, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat((pos_encoding, metadata), dim=-1)  # [B L D]

        patches = patches + pos_metadata_encoding  # [B L D] + [B L D] -> [B L D]
        return patches  # [B L D]

    def mask_out(self, patches):
        """
        Mask out patches randomly by shuffling the patches & masking out the
        first N patches

        Parameters
        ----------
        patches : torch.Tensor A tensor of shape (B, L, D)

        Returns
        -------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, L:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.
        masked_matrix : torch.Tensor
            A tensor of shape (B, L) containing the mask matrix, 1 indicates a masked patch & 0 indicates an unmasked patch.
        """
        B, L, D = patches.shape
        # assert (
        #     L == self.num_patches
        # ), f"Expected {self.num_patches} patches, got {L} patches."

        if self.training:  # Shuffle the patches
            noise = torch.randn((B, L), device=patches.device)  # [B L]
        else:  # Don't shuffle, useful for interpolation & inspection of embeddings
            noise = rearrange(
                torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L
            )

        random_indices = torch.argsort(noise, dim=-1)  # [B L]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B L]

        num_masked_patches = int(
            self.mask_ratio * self.num_patches
        )  # Number of patches to be masked out
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],  # [B mask_ratio * L]
            random_indices[:, num_masked_patches:],  # [B (1 - mask_ratio) * L]
        )

        # create a mask of shape B L, where 1 indicates a masked patch
        # and 0 indicates an unmasked patch
        masked_matrix = torch.zeros((B, L), device=patches.device)  # [B L] = 0
        masked_matrix[:, :num_masked_patches] = 1  # [B mask_ratio * L] = 1
        masked_matrix = torch.gather(
            masked_matrix, dim=1, index=reverse_indices
        )  # [B L] -> [B L] - reorder the patches

        # mask out the patches
        batch_indices = rearrange(
            torch.arange(B, device=patches.device), "B -> B 1"
        )  # [B 1]
        unmasked_patches = patches[
            batch_indices, unmasked_indices, :
        ]  # [B L:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B L:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # [B, 1]
            datacube["waves"],  # [B N]
        )  # [B C H W]

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(
            cube, waves[0]
        )  # [B L D] - patchify & create embeddings per patch
        # TODO: Add time & latlon as encoding to patches
        patches = self.add_encodings(
            patches,
            time,
            latlon,
            gsd,
        )  # [B L D] - add position encoding to the embeddings

        # mask out patches
        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.mask_out(
            patches
        )  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        unmasked_patches = torch.cat(
            (cls_tokens, unmasked_patches), dim=1
        )  # [B (1 + L) D]

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches = self.transformer(
            unmasked_patches
        )  # [B ((1 + L)):(1 - mask_ratio)) D]

        return (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B ((1 + L):(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B L]


class Decoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        encoder_dim,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.dim = dim

        self.enc_to_dec = (
            nn.Linear(encoder_dim, dim) if encoder_dim != dim else nn.Identity()
        )
        self.mask_patch = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
        )
        self.embed_to_pixels = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=True,
        )

    def reconstruct_and_add_encoding(
        self,
        unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
        time,
        latlon,
        gsd,
    ):
        B, L = masked_matrix.shape
        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2
        cls_tokens, unmasked_patches = (
            unmasked_patches[:, :1, :],
            unmasked_patches[:, 1:, :],
        )  # [B 1 D], [B L:(1 - mask_ratio) D]

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size, w=grid_size, dim=(self.dim - 8), gsd=gsd[0]
            )
            .to(unmasked_patches.device)
            .detach()
        )  # [L D]
        metadata = (
            encode_metadata(time, latlon, has_time=True, has_lat_lon=True)
            .to(unmasked_patches.device)
            .detach()
        )  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        metadata = repeat(metadata, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat((pos_encoding, metadata), dim=-1)  # [B L D]

        batch_indices = rearrange(
            torch.arange(B, device=unmasked_patches.device), "B -> B 1"
        )  # [B 1]

        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_patches = repeat(
            self.mask_patch, "D -> B L D", B=B, L=num_masked_patches
        )  # [B L:mask_ratio D]

        # Add position encoding
        masked_patches = (
            masked_patches + pos_metadata_encoding[batch_indices, masked_indices, :]
        )  # [B L:mask_ratio D] + [B L:mask_ratio D]
        unmasked_patches = (
            unmasked_patches + pos_metadata_encoding[batch_indices, unmasked_indices, :]
        )  # [B GL:(1 - masked_ratio) D] + [B GL:(1 - mask_ratio) D]

        # Concatenate the masked & unmasked patches
        decoder_patches = torch.zeros(
            (B, self.num_patches, self.dim), device=unmasked_patches.device
        )  # [B L D]
        decoder_patches[batch_indices, unmasked_indices, :] = (
            unmasked_patches  # [B L:(1 - mask_ratio) D])
        )
        decoder_patches[batch_indices, masked_indices, :] = (
            masked_patches  # [B L:mask_ratio D])
        )

        decoder_patches = torch.cat(
            (cls_tokens, decoder_patches), dim=1
        )  # [B (1 + L) D]

        return decoder_patches  # [B (1 + L) D]

    def pixelify(self, patches):
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

    def forward(
        self,
        encoded_unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
        time,
        latlon,
        gsd,
        waves,
    ):
        # Change the embedding dimension from encoder to decoder
        encoded_unmasked_patches = self.enc_to_dec(
            encoded_unmasked_patches
        )  # [B (1 + L) D]

        # Reconstruct the patches to feed into the decoder transformer
        decoder_patches = self.reconstruct_and_add_encoding(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
            time,
            latlon,
            gsd[0],
        )  # [B (1 + L) D]

        # Pass the decoder patches through the transformer
        decoded_patches = self.transformer(decoder_patches)  # [B (1 + L) D]

        pixels, waves = self.embed_to_pixels(
            decoded_patches, waves[0]
        )  # [B (1 + L) (C P P)]
        # Remove the class token
        pixels = pixels[:, 1:, :]
        return pixels, waves  # [B L (C P P)], [B N]


class ClayMAE(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        norm_pix_loss,
        # ENCODER
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        # DECODER
        decoder_dim,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        decoder_mlp_ratio,
        **kwargs,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
        )

        self.decoder = Decoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            encoder_dim=dim,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_ratio=decoder_mlp_ratio,
        )

    def per_pixel_loss(self, cube, pixels, masked_matrix):
        """
        cube: [B C H W]
        pixels: [B L (C P P)]
        masked_matrix: [B L], 0 is unmasked, 1 is masked
        """
        patches = rearrange(
            cube,
            "B C (h p1) (w p2) -> B (h w) (C p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )  # [B L (C P P)]

        if self.norm_pix_loss:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            patches = (patches - mean) / (var + 1e-6) ** 0.5

        loss = F.mse_loss(patches, pixels, reduction="none")  # loss per pixel
        loss = reduce(loss, "B L D -> B L", reduction="mean")  # loss per patch

        loss = (
            loss * masked_matrix
        ).sum() / masked_matrix.sum()  # loss on masked patches only

        return loss

    def forward(self, datacube):
        """
        datacube: dict containing the following keys:
            - pixels: [B C H W]
            - time: [B 2]
            - latlon: [B 2]
            - gsd: [B 1]
            - waves: [B N]
        """
        # ENCODER
        (
            encoded_unmasked_patches,  # [B (1 + L):(1 - mask_ratio) D]
            unmasked_indices,  # [(1-mask_ratio)]
            masked_indices,  # [mask_ratio]
            masked_matrix,  # [B L]
        ) = self.encoder(datacube)

        # DECODER
        pixels, waves = self.decoder(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
            datacube["time"],
            datacube["latlon"],
            datacube["gsd"],
            datacube["waves"],
        )  # [B L (C P P)]

        # LOSS
        loss = self.per_pixel_loss(datacube["pixels"], pixels, masked_matrix)

        return loss
