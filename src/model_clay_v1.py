import math
import os
import re
from typing import Literal

import geopandas as gpd
import lightning as L
import numpy as np
import pandas as pd
import pyarrow as pa
import shapely
import timm
import torch
import torch.nn.functional as F
import yaml
from box import Box
from einops import rearrange, reduce, repeat
from torch import nn
from torchvision.transforms import v2
from vit_pytorch.simple_vit import Transformer

from src.factory import DynamicEmbedding
from src.utils import posemb_sincos_2d_with_gsd

torch.set_float32_matmul_precision("medium")
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


class Encoder(nn.Module):
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
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
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
                gsd=gsd,
            )
            .to(patches.device)
            .detach()
        )  # [L (D - 8)]

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat(
            (pos_encoding, time_latlon), dim=-1
        )  # [B L D]

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
            A tensor of shape (B, L) containing the mask matrix, 1 indicates a masked
            patch & 0 indicates an unmasked patch.
        """
        B, L, D = patches.shape
        # assert (
        #     L == self.num_patches
        # ), f"Expected {self.num_patches} patches, got {L} patches."

        if self.shuffle:  # Shuffle the patches
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
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )  # [B C H W]

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(
            cube, waves
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

    def reconstruct_and_add_encoding(  # noqa: PLR0913
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
                h=grid_size, w=grid_size, dim=(self.dim - 8), gsd=gsd
            )
            .to(unmasked_patches.device)
            .detach()
        )  # [L D]
        time_latlon = (
            torch.hstack((time, latlon)).to(unmasked_patches.device).detach()
        )  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat(
            (pos_encoding, time_latlon), dim=-1
        )  # [B L D]

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

    def forward(  # noqa: PLR0913
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
            gsd,
        )  # [B (1 + L) D]

        # Pass the decoder patches through the transformer
        decoded_patches = self.transformer(decoder_patches)  # [B (1 + L) D]

        pixels, waves = self.embed_to_pixels(
            decoded_patches, waves
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
        shuffle,
        metadata,
        teacher,
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
        self.shuffle = shuffle
        self.metadata = metadata
        self.teacher = timm.create_model(teacher, pretrained=True, num_classes=0)
        self.teacher_chip_size = 224
        self.teacher_resize = v2.Resize(
            size=(self.teacher_chip_size, self.teacher_chip_size)
        )
        self.proj = nn.Linear(dim, self.teacher.num_features)

        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            shuffle=shuffle,
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

        self.freeze_teacher()

    def freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad = False

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

        loss = F.l1_loss(patches, pixels, reduction="none")  # loss per pixel
        loss = reduce(loss, "B L D -> B L", reduction="mean")  # loss per patch

        loss = (
            loss * masked_matrix
        ).sum() / masked_matrix.sum()  # loss on masked patches only

        return loss

    def forward(self, datacube):
        """
        datacube: dict containing the following keys:
            - pixels: [B C H W]
            - time: [B 4] # week hour
            - latlon: [B 4] # lat lon
            - platform: [B 1]
            - date: [B 1]
        """
        platform = datacube["platform"][0]
        waves = torch.tensor(list(self.metadata[platform].bands.wavelength.values()))
        gsd = torch.tensor(self.metadata[platform].gsd)

        # ENCODER
        (
            encoded_unmasked_patches,  # [B (1 + L):(1 - mask_ratio) D]
            unmasked_indices,  # [(1-mask_ratio)]
            masked_indices,  # [mask_ratio]
            masked_matrix,  # [B L]
        ) = self.encoder(
            {
                "pixels": datacube["pixels"],
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            }
        )

        # DECODER
        pixels, waves = self.decoder(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
            datacube["time"],
            datacube["latlon"],
            gsd,
            waves,
        )  # [B L (C P P)]

        # LOSS
        reconstruction_loss = self.per_pixel_loss(
            datacube["pixels"], pixels, masked_matrix
        )

        # TEACHER
        encoder_output = self.proj(encoded_unmasked_patches[:, 0, :])  # [B D']
        with torch.no_grad():
            if platform == "sentinel-1-rtc":
                r = datacube["pixels"][:, 0, :, :]
                g = datacube["pixels"][:, 1, :, :]
                b = r - g
                rgb = torch.stack((r, g, b), dim=1)
            else:
                # Read RGB bands from the sensor to feed the teacher model
                indices = self.metadata[platform].rgb_indices
                rgb = datacube["pixels"][:, indices, :, :]
            rgb = self.teacher_resize(rgb)
            teacher_output = self.teacher(rgb)

        representation_loss = -(
            F.cosine_similarity(encoder_output, teacher_output).mean()
            - 1.0  # change range from [-1, 1] to [-2, 0]
        )  # negative cosine similarity, [0, 2] -> 0 is similar & 2 is opposite

        loss = 0.90 * reconstruction_loss + 0.10 * representation_loss
        return (loss, reconstruction_loss, representation_loss)


def clay_mae_tiny(**kwargs):
    args = {
        # ENCODER
        "dim": 192,
        "depth": 6,
        "heads": 4,
        "dim_head": 48,
        "mlp_ratio": 2,
        # DECODER
        "decoder_dim": 96,
        "decoder_depth": 3,
        "decoder_heads": 2,
        "decoder_dim_head": 48,
        "decoder_mlp_ratio": 2,
    }
    args.update(kwargs)
    return ClayMAE(**args)


def clay_mae_small(**kwargs):
    args = {
        # ENCODER
        "dim": 384,
        "depth": 6,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 2,
        # DECODER
        "decoder_dim": 192,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 2,
    }
    args.update(kwargs)
    return ClayMAE(**args)


def clay_mae_base(**kwargs):
    args = {
        # ENCODER
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        # DECODER
        "decoder_dim": 512,
        "decoder_depth": 6,
        "decoder_heads": 6,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return ClayMAE(**args)


def clay_mae_large(**kwargs):
    args = {
        # ENCODER
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        # DECODER
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 8,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return ClayMAE(**args)


class ClayMAEModule(L.LightningModule):
    def __init__(  # noqa: PLR0913
        self,
        model_size="base",
        mask_ratio=0.75,
        norm_pix_loss=False,
        patch_size=16,
        shuffle=False,
        metadata_path="configs/metadata.yaml",
        teacher="vit_base_patch16_224.dino",
        lr=1e-4,
        wd=0.05,
        b1=0.9,
        b2=0.95,
        embeddings_level: Literal["mean", "patch", "group"] = "mean",
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        model_map = {
            "tiny": clay_mae_tiny,
            "small": clay_mae_small,
            "base": clay_mae_base,
            "large": clay_mae_large,
        }
        if model_size in model_map:
            model_args = {
                "mask_ratio": mask_ratio,
                "patch_size": patch_size,
                "norm_pix_loss": norm_pix_loss,
                "shuffle": shuffle,
                "metadata": self.metadata,
                "teacher": teacher,
            }
            self.model = model_map[model_size](**model_args)
        else:
            raise ValueError(
                f"Invalid model size {model_size}. Expected one of {model_map.keys()}"
            )

    def on_train_epoch_start(self):
        self.model.teacher.eval()

    def forward(self, datacube: dict[str, torch.Tensor]):
        return self.model(datacube)

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
        datacube = batch
        loss, reconstruction_loss, representation_loss = self(datacube)
        self.log(
            name=f"{phase}/loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            name=f"{phase}/rec_loss",
            value=reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            name=f"{phase}/rep_loss",
            value=representation_loss,
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
            [
                self.model.encoder.B,
                256,  # (batch_size, seq_length, hidden_size),
                768,  # assume image size is 256 x 256 & patch size is 16 x16
            ]
        )
        assert not torch.isnan(embeddings_raw).any()  # ensure no NaNs in embedding

        if self.hparams.embeddings_level == "mean":
            # Take the mean of the embeddings along the sequence_length dimension
            # i.e. compute mean over patch embeddings only
            embeddings_output = reduce(embeddings_raw, "b l d -> b d", "mean")
            expected_size = [self.model.encoder.B, 768]  # (batch_size, hidden_size)
        elif self.hparams.embeddings_level in "patch":
            # Rearrange the raw embeddings into h x w.
            embeddings_output = rearrange(
                embeddings_raw, "b (h w) d -> b h w d", w=16, h=16
            )
            expected_size = [
                self.model.encoder.B,
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
