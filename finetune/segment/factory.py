import re

import torch
from einops import rearrange, repeat
from torch import nn

from src.model import Encoder


class SegmentEncoder(Encoder):
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
        feature_maps=[2, 4],
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
        self.feature_maps = feature_maps

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fpn5 = nn.Identity()

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.load_from_ckpt(ckpt_path)

    def load_from_ckpt(self, ckpt_path):
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
        patches = self.add_encodings(
            patches,
            time,
            latlon,
            gsd,
        )  # [B L D] - add position encoding to the embeddings

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        features = []
        for idx, (attn, ff) in enumerate(self.transformer.layers):
            patches = attn(patches) + patches
            patches = ff(patches) + patches
            if idx in self.feature_maps:
                _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=28, W=28)
                features.append(_cube)
        patches = self.transformer.norm(patches)
        _cube = rearrange(patches[:, 1:, :], "B (H W) D -> B D H W", H=28, W=28)
        features.append(_cube)

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4, self.fpn5]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return features


class Segmentor(nn.Module):
    def __init__(self, num_classes, feature_maps, ckpt_path):
        super().__init__()
        self.encoder = SegmentEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=768,
            depth=12,
            heads=12,
            dim_head=64,
            mlp_ratio=4.0,
            feature_maps=feature_maps,
            ckpt_path=ckpt_path,
        )
        self.upsamples = [nn.Upsample(scale_factor=2**i) for i in range(4)] + [
            nn.Upsample(scale_factor=4)
        ]
        self.fusion = nn.Conv2d(self.encoder.dim * 5, self.encoder.dim, kernel_size=1)
        self.seg_head = nn.Conv2d(self.encoder.dim, num_classes, kernel_size=1)

    def forward(self, datacube):
        features = self.encoder(datacube)
        for i in range(len(features)):
            features[i] = self.upsamples[i](features[i])

        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)

        logits = self.seg_head(fused)
        return logits
