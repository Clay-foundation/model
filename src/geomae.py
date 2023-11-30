import torch
from einops import rearrange, reduce, repeat
from torch import nn
from vit_pytorch.vit import Transformer
from src.utils import posemb_sincos_2d, posemb_sincos_1d


class Patchify(nn.Module):
    """Patchify the input cube & create embeddings per patch

    Args:
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension
        patch_size (int): Patch size
    """

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


class Encoder(nn.Module):
    def __init__(
        self,
        mask_ratio,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        dim_head,
        bands,
        band_groups,
        dropout,
        emb_dropout,
        device,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.bands = bands
        self.band_groups = band_groups
        self.device = device
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_group_patches = len(band_groups)
        self.num_patches = self.num_spatial_patches * self.num_group_patches
        self.num_masked_patches = int(self.mask_ratio * self.num_patches)

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
        # self.pos_encoding = nn.Embedding(
        #     num_embeddings=self.num_spatial_patches, embedding_dim=pos_dim
        # )
        # self.band_encoding = nn.Embedding(
        #     num_embeddings=self.num_group_patches, embedding_dim=band_dim
        # )
        self.pos_encoding = posemb_sincos_2d(
            h=image_size // patch_size, w=image_size // patch_size, dim=pos_dim
        )  # [L D/2]
        self.band_encoding = posemb_sincos_1d(
            length=self.num_group_patches, dim=band_dim
        )  # [G D/2]

        # freeze the position & band encoding
        self.pos_encoding = self.pos_encoding.to(self.device).requires_grad_(False)
        self.band_encoding = self.band_encoding.to(self.device).requires_grad_(False)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * 2,
            dropout=dropout,
        )
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Linear(dim, num_classes)

    def to_patch_embed(self, cube):
        """Patchify the input cube & create embeddings per patch

        Args:
            cube (torch.Tensor): A tensor of shape (B, C, H, W) containing the pixels of the datacube

        Returns:
            patches (torch.Tensor): A tensor of shape (B, G, L, D) containing the embeddings of the patches
        """
        patches = []
        for name, bands in self.band_groups.items():
            cubeslice = cube[:, bands, :, :]  # [B C H W] -> [B C[slice[...]] H W]
            patches.append(self.patch_embedding[name](cubeslice))

        patches = rearrange(patches, "G B L D -> B G L D")  # [B G L D]
        return patches  # [B G L D]

    def add_encodings(self, patches):
        """Add position & band encoding to the patches

        Args:
            patches (torch.Tensor): A tensor of shape (B, G, L, D) containing the embeddings of the patches

        Returns:
            patches (torch.Tensor): A tensor of shape (B, G, L, D) containing the embeddings of the patches + position & band encoding
        """
        B, G, L, D = patches.shape

        # align position & band embeddings across patches
        pos_encoding = repeat(
            self.pos_encoding, "L D -> 1 repeat L D", repeat=G
        )  # [1 G L D/2]

        band_encoding = repeat(
            self.band_encoding, "G D -> 1 G repeat D", repeat=L
        )  # [1 G L D/2]
        # pos_encoding = repeat(
        #     rearrange(
        #         self.pos_encoding(torch.arange(L, device=patches.device)),
        #         "L D -> 1 1 L D",
        #     ),
        #     "1 1 L D -> 1 repeat L D",
        #     repeat=G,
        # )  # [1 G L D/2]
        # band_encoding = repeat(
        #     rearrange(
        #         self.band_encoding(torch.arange(G, device=patches.device)),
        #         "G D -> 1 G 1 D",
        #     ),
        #     "1 G 1 D -> 1 G repeat D",
        #     repeat=L,
        # )  # [1 G L D/2]

        pos_band_encoding = torch.cat(
            (pos_encoding, band_encoding), dim=-1
        )  # [1 G L D]

        # add position & band encoding to the input feature vector
        patches = patches + pos_band_encoding  # [B G L D] + [1 G L D] - broadcasting
        patches = self.dropout(patches)  # [B G L D]
        return patches  # [B G L D]

    def embed_metadata(self, patches, latlon, time):
        """Add timestep & latlon embedding to the patches

        Args:
            patches (torch.Tensor): A tensor of shape (B, GL, D) containing the embeddings of the patches + position & band encoding
            latlon (torch.Tensor): A tensor of shape (B, 2) containing the latlon of the datacube
            time (torch.Tensor): A tensor of shape (B, 2) containing the timestep of the datacube

        Returns:
            patches (torch.Tensor): A tensor of shape (B, GL, D) containing the embeddings of the patches + position & band encoding + timestep & latlon embedding
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
        """Mask out patches randomly by shuffling the patches & masking out the first N patches

        Args:
            patches (torch.Tensor): A tensor of shape (B, GL, D) containing the embeddings of the patches + position & band encoding + timestep & latlon embedding

        Returns:
            unmasked_patches (torch.Tensor): A tensor of shape (B, GL:(1 - mask_ratio), D) containing the embeddings of the unmasked patches
            unmasked_indices (torch.Tensor): A tensor of shape (B, (1 - mask_ratio)) containing the indices of the unmasked patches
            masked_indices (torch.Tensor): A tensor of shape (B, mask_ratio) containing the indices of the masked patches
            masked_matrix (torch.Tensor): A tensor of shape (B, G, L) containing the mask matrix
        """
        B, GL, D = patches.shape
        assert (
            GL == self.num_patches
        ), f"Expected {self.num_patches} patches, got {GL} patches."

        noise = torch.randn((B, GL), device=patches.device)  # [B GL]
        random_indices = torch.argsort(noise, dim=-1)  # [B GL]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B GL]

        masked_indices, unmasked_indices = (
            random_indices[:, : self.num_masked_patches],  # [B mask_ratio * GL]
            random_indices[:, self.num_masked_patches :],  # [B (1 - mask_ratio) * GL]
        )

        # create a mask of shape B G L, where 1 indicates a masked patch & 0 indicates an unmasked patch
        masked_matrix = torch.zeros((B, GL), device=patches.device)  # [B GL] = 0
        masked_matrix[:, : self.num_masked_patches] = 1  # [B mask_ratio * GL] = 1
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
        masked_patches = patches[
            batch_indices, masked_indices, :
        ]  # [B GL:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B GL:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B G L]

    def forward(self, datacube):
        """"""
        # pdb.set_trace()
        cube, time, latlon = (
            datacube["pixels"],
            datacube["timestep"],
            datacube["latlon"],
        )  # [B C H W], [B 2], [B 2]

        B, C, H, W = cube.shape

        patches = self.to_patch_embed(
            cube
        )  # [B G L D] - patchify & create embeddings per patch
        # print("patches", patches.mean(dim=(0, 2, 3)))

        patches = self.add_encodings(
            patches
        )  # [B G L D] - add position & band encoding to the embeddings
        # print("patches + encode", patches.mean(dim=(0, 2, 3)))

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
        # print("unmasked_patches", unmasked_patches.mean(dim=(0, 2)))

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches = self.transformer(
            unmasked_patches
        )  # [B (GL:(1 - mask_ratio) + 2) D]
        # print("encoded_unmasked_patches", encoded_unmasked_patches.mean(dim=(0, 2)))

        return (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B (GL:(1 - mask_ratio) + 2) D], [(1-mask_ratio)], [mask_ratio], [B G L]


class Decoder(nn.Module):
    def __init__(
        self,
        mask_ratio,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        dim_head,
        bands,
        band_groups,
        dropout,
        device,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.band_groups = band_groups
        self.device = device
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_group_patches = len(band_groups)
        self.num_patches = self.num_spatial_patches * self.num_group_patches
        self.num_masked_patches = int(self.mask_ratio * self.num_patches)

        self.mask_patch = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * 2,
            dropout=dropout,
        )

        pos_dim = band_dim = dim // 2

        self.pos_encoding = posemb_sincos_2d(
            h=image_size // patch_size, w=image_size // patch_size, dim=pos_dim
        )  # [L D/2]
        self.band_encoding = posemb_sincos_1d(
            length=self.num_group_patches, dim=band_dim
        )  # [G D/2]
        # self.pos_encoding = nn.Embedding((image_size // patch_size) ** 2, pos_dim)
        # self.band_encoding = nn.Embedding(len(band_groups), band_dim)

        # freeze the position & band encoding
        self.pos_encoding = self.pos_encoding.to(self.device).requires_grad_(False)
        self.band_encoding = self.band_encoding.to(self.device).requires_grad_(False)

        self.embed_to_pixels = nn.ModuleDict(
            {
                name: nn.Linear(dim, len(bands) * (patch_size**2))
                for name, bands in self.band_groups.items()
            }
        )

    def reconstruct_and_add_encoding(
        self, unmasked_patches, unmasked_indices, masked_indices
    ):
        """Reconstruct the input patches from the random mask patch & add position & band encoding to them

        Args:
            unmasked_patches (torch.Tensor): A tensor of shape (B, GL:(1 - mask_ratio), D) containing the embeddings of the unmasked patches
            unmasked_indices (torch.Tensor): A tensor of shape (B, (1 - mask_ratio)) containing the indices of the unmasked patches
            masked_indices (torch.Tensor): A tensor of shape (B, mask_ratio) containing the indices of the masked patches

        Returns:
            decoder_patches (torch.Tensor): A tensor of shape (B, GL, D) containing the embeddings for the decoder part of the model
        """
        import pdb

        pdb.set_trace()
        B, *_ = unmasked_patches.shape

        # align position & band embeddings across patches
        pos_encoding = repeat(
            self.pos_encoding, "L D -> 1 repeat L D", repeat=self.num_group_patches
        )  # [1 G L D/2]
        band_encoding = repeat(
            self.band_encoding, "G D -> 1 G repeat D", repeat=self.num_spatial_patches
        )  # [1 G L D/2]
        # pos_encoding = repeat(
        #     rearrange(
        #         self.pos_encoding(
        #             torch.arange(
        #                 self.num_spatial_patches, device=unmasked_patches.device
        #             )
        #         ),
        #         "L D -> 1 1 L D",
        #     ),
        #     "1 1 L D -> 1 repeat L D",
        #     repeat=self.num_group_patches,
        # )  # [1 G L D/2]
        # band_encoding = repeat(
        #     rearrange(
        #         self.band_encoding(
        #             torch.arange(self.num_group_patches, device=unmasked_patches.device)
        #         ),
        #         "G D -> 1 G 1 D",
        #     ),
        #     "1 G 1 D -> 1 G repeat D",
        #     repeat=self.num_spatial_patches,
        # )  # [1 G L D/2]

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

        # reconstruct the masked patches from the random mask patch & add position & band encoding to them
        masked_patches = repeat(
            self.mask_patch, "D -> B GL D", B=B, GL=self.num_masked_patches
        )  # [B GL:mask_ratio D]
        masked_patches = (
            masked_patches + masked_pos_band_encoding
        )  # [B GL:mask_ratio D] + [B GL:mask_ratio D]

        # add position & band encoding to the unmasked patches
        unmasked_patches = (
            unmasked_patches + unmasked_pos_band_encoding
        )  # [B GL:(1 - masked_ratio) D] + [B GL:(1 - mask_ratio) D]

        decoder_patches = torch.zeros(
            (B, self.num_patches, self.dim), device=unmasked_patches.device
        )  # [B GL D]
        decoder_patches[
            batch_indices, unmasked_indices, :
        ] = unmasked_patches  # [B GL:(1 - mask_ratio) D]
        decoder_patches[
            batch_indices, masked_indices, :
        ] = masked_patches  # [B GL:mask_ratio D]

        return decoder_patches  # [B GL D]

    def pixelify(self, patches):
        """Convert the patches into pixel space to compute the loss

        Args:
            patches (torch.Tensor): A tensor of shape (B, GL, D) containing the embeddings from the decoder part of the model

        Returns:
            pixels (torch.Tensor): A tensor of shape (B, C, L, PP) containing the pixels of the datacube
        """
        patches = rearrange(
            patches, "B (G L) D -> B G L D", G=len(self.band_groups)
        )  # [B G L D]
        pixels = []
        for i, (name, bands) in enumerate(self.band_groups.items()):
            group_embeddings = patches[:, i, :, :]  # [B L D]
            group_pixels = self.embed_to_pixels[name](group_embeddings)  # [B L (C P P)]
            group_pixels = rearrange(
                group_pixels,
                "B L (C PP) -> B C L PP",
                PP=(self.patch_size**2),
            )  # [B C L PP]
            pixels.append(group_pixels)  # [B C L PP]

        pixels = torch.cat(pixels, dim=1)  # [B C L PP]
        return pixels  # [B C L PP]

    def forward(self, encoded_unmasked_patches, unmasked_indices, masked_indices):
        encoded_unmasked_patches, encoded_unmasked_meta_patches = (
            encoded_unmasked_patches[:, :-2, :],
            encoded_unmasked_patches[:, -2:, :],
        )  # [B (GL:(1 - mask_ratio)) D], [B 2 D]

        # reconstruct the patches to feed into the decoder transformer
        decoder_patches = self.reconstruct_and_add_encoding(
            encoded_unmasked_patches, unmasked_indices, masked_indices
        )  # [B GL D]

        # add the metadata patches to the decoder patches
        decoder_patches = torch.cat(
            [decoder_patches, encoded_unmasked_meta_patches], dim=1
        )  # [B (GL + 2) D]

        # pass the decoder patches through the transformer
        decoded_patches = self.transformer(decoder_patches)  # [B (GL + 2) D]

        # remove the metadata patches from the decoded patches
        decoded_patches = decoded_patches[:, :-2, :]  # [B GL D]

        # pixelify the decoded patches
        pixels = self.pixelify(decoded_patches)  # [B C L PP]
        return pixels


class GeoMAE(nn.Module):
    def __init__(
        self,
        mask_ratio=0.75,
        image_size=256,
        patch_size=32,
        bands=13,
        band_groups={
            "rgb": (2, 1, 0),
            "rededge": (3, 4, 5, 7),
            "nir": (6,),
            "swir": (8, 9),
            "sar": (10, 11),
            "dem": (12,),
        },
        # ENCODER
        dim=128,
        depth=2,
        heads=4,
        dim_head=32,
        dropout=0.0,
        emb_dropout=0.0,
        # DECODER
        decoder_dim=128,
        decoder_depth=1,
        decoder_heads=4,
        decoder_dim_head=32,
        decoder_dropout=0.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.bands = bands
        self.band_groups = band_groups
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device", device)

        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            bands=bands,
            band_groups=band_groups,
            dropout=dropout,
            emb_dropout=emb_dropout,
            device=device,
        )

        self.decoder = Decoder(
            mask_ratio=mask_ratio,
            image_size=image_size,
            patch_size=patch_size,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            bands=bands,
            band_groups=band_groups,
            dropout=decoder_dropout,
            device=device,
        )

    def per_pixel_loss(self, cube, pixels, masked_matrix):
        """Compute the per pixel loss

        Args:
            cube (torch.Tensor): A tensor of shape (B, C, H, W) containing the pixels of the datacube
            pixels (torch.Tensor): A tensor of shape (B, C, L, PP) containing the pixels per patch of the datacube
            masked_matrix (torch.Tensor): A tensor of shape (B, G, L) containing the mask matrix

        Returns:
            loss"""
        patches = rearrange(
            cube,
            "B C (h p1) (w p2) -> B C (h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )  # [B C L PP]
        # print("patch", patches.mean(dim=(0, 2, 3)))
        # print("pixel", pixels.mean(dim=(0, 2, 3)))
        loss = (patches - pixels) ** 2  # loss per pixel
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

        # print(actual_loss, masked_patches_in_group)
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
        # print("encoded_unmasked_patches", encoded_unmasked_patches.mean(dim=(0, 2)))

        # DECODER
        pixels = self.decoder(
            encoded_unmasked_patches, unmasked_indices, masked_indices
        )  # [B C L PP]

        # LOSS
        loss = self.per_pixel_loss(datacube["pixels"], pixels, masked_matrix)

        return loss


if __name__ == "__main__":
    import math

    # Random data
    cube = {
        "pixels": torch.randn(3, 13, 256, 256),
        "timestep": torch.tensor(
            data=[[2017.0, 10.0, 5.0], [2020.0, 1.0, 31.0], [2022.0, 5.0, 15.0]]
        ),
        "latlon": torch.tensor(
            data=[
                [math.radians(57.0), math.radians(-2.0)],
                [math.radians(27.0), math.radians(16.5)],
                [math.radians(57.0), math.radians(-2.0)],
            ]
        ),
    }

    model = GeoMAE()
    loss = model(cube)
    # print(loss)
