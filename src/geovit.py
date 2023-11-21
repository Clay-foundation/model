import math

import torch
from torch import nn
from einops import rearrange, repeat
from vit_pytorch.vit import Transformer


class Patchify(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size):
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


class GeoViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        bands=12,
        band_groups={
            "rgb": (2, 3, 4),
            "nir": (0, 1),
            "swir": (5, 7, 8),
            "sar": (6, 9),
            "dem": (10, 11),
        },
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_size // patch_size) ** 2

        self.band_groups = band_groups
        pos_dim = band_dim = dim // 2

        self.latlon_embedding = nn.Linear(2, dim)
        self.time_embedding = nn.Linear(2, dim)
        self.patch_embedding = nn.ModuleDict(
            {
                name: Patchify(len(bands), dim, patch_size)
                for name, bands in self.band_groups.items()
            }
        )
        self.pos_encoding = nn.Embedding(
            num_embeddings=num_patches, embedding_dim=pos_dim
        )
        self.band_encoding = nn.Embedding(
            num_embeddings=len(self.band_groups), embedding_dim=band_dim
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, datacube):
        cube, timestep, latlon = (
            datacube["pixels"],
            datacube["timestep"],
            datacube["latlon"],
        )
        B, C, H, W = cube.shape

        # patchify & create embeddings per patch
        patches = []
        for name, bands in self.band_groups.items():
            cubeslice = cube[:, bands, :, :]
            patches.append(self.patch_embedding[name](cubeslice))

        x = rearrange(patches, "G B L D -> B G L D")
        B, G, L, D = x.shape

        # align position & band embeddings across patches
        pos_encoding = repeat(
            rearrange(self.pos_encoding(torch.arange(L)), "L D -> 1 1 L D"),
            "1 1 L D -> 1 repeat L D",
            repeat=G,
        )
        band_encoding = repeat(
            rearrange(self.band_encoding(torch.arange(G)), "G D -> 1 G 1 D"),
            "1 G 1 D -> 1 G repeat D",
            repeat=L,
        )

        pos_band_encoding = torch.cat((pos_encoding, band_encoding), dim=-1)

        # add position & band encoding to the input feature vector
        x = x + pos_band_encoding
        x = rearrange(x, "B G L D -> B (G L) D")
        x = self.dropout(x)

        # add timestep & latlon embedding
        latlon_embedding = rearrange(self.latlon_embedding(latlon), "B D -> B 1 D")
        time_embedding = rearrange(self.time_embedding(timestep), "B D -> B 1 D")
        x = torch.cat([x, latlon_embedding, time_embedding], dim=1)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.to_latent(x)

        return x
        # return self.mlp_head(x)


def main():
    # model
    g = GeoViT(
        image_size=256,
        patch_size=32,
        num_classes=2,
        dim=128,
        depth=2,
        heads=8,
        mlp_dim=256,
        bands=12,
        dropout=0.1,
        emb_dropout=0.1,
    )

    # sample data
    cube = {
        "pixels": torch.randn(1, 12, 256, 256),
        "timestep": torch.tensor(data=[24.0, 13.0]).unsqueeze(0),
        "latlon": torch.tensor(data=[math.radians(57.0), math.radians(-2.0)]).unsqueeze(
            0
        ),
    }

    preds = g(cube)
    print(preds.shape)


if __name__ == "__main__":
    main()
