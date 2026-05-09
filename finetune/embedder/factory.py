"""Export the Clay model to ONNX and pytorch ExportedProgram format.

This script exports the Clay model to ONNX and pytorch ExportedProgram format
for deployment. The model is exported with dynamic shapes for inference.

How to use:

```bash
python -m finetune.embedder.factory \
    --img_size 256 \
    --ckpt_path checkpoints/clay_v1.5.ckpt \
    --device cuda \
    --name clay-v1.5-encoder.onnx \
    --onnx
# exports Clay encoder to ONNX format that can handle chips of size 256x256
# for different sensors like Sentinel-2, Landsat-8, NAIP, LINZ & Sentinel 1.
```

```bash
python -m finetune.embedder.factory \
    --img_size 224 \
    --ckpt_path checkpoints/clay_v1.5.ckpt \
    --device cuda \
    --name clay-v1.5-encoder.pt2 \
    --ep
# exports Clay encoder to pytorch ExportedProgram format that can handle chips
# of size 224x224 for different sensors like Sentinel-2, Landsat-8, NAIP, LINZ
# & Sentinel 1.
```

"""

import argparse
import warnings
from pathlib import Path
from typing import Any

import torch
from einops import repeat
from torch import nn
from torch.export import Dim

from claymodel.model import Encoder
from claymodel.utils import load_encoder_weights, posemb_sincos_2d_with_gsd

warnings.filterwarnings("ignore", category=UserWarning)


class EmbeddingEncoder(Encoder):
    """Clay Encoder without mask and shuffle."""

    def __init__(  # noqa: PLR0913
        self,
        img_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__(
            mask_ratio=0.0,
            shuffle=False,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
        )
        self.img_size = img_size

        # Using fixed grid size for inference
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

    def add_encodings(
        self,
        patches: torch.Tensor,
        time: torch.Tensor,
        latlon: torch.Tensor,
        gsd: torch.Tensor,
    ) -> torch.Tensor:
        """Add position encoding to the patches"""
        B, L, _ = patches.shape

        grid_size = self.grid_size

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
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)  # [B L D]

        return patches + pos_metadata_encoding  # [B L D] + [B L D] -> [B L D]

    def forward(self, datacube: dict[str, torch.Tensor]) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )  # [B C H W]
        B = cube.shape[0]

        patches, _ = self.to_patch_embed(
            cube, waves
        )  # [B L D] - patchify & create embeddings per patch

        # Add time & latlon as encoding to patches
        patches = self.add_encodings(
            patches,
            time,
            latlon,
            gsd,
        )  # [B L D] - add position encoding to the embeddings

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        # pass the patches through the transformer
        patches = self.transformer(patches)  # [B (1 + L) D]

        # get the cls token
        return patches[:, 0, :]  # [B D]


class Embedder(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        ckpt_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.clay_encoder = EmbeddingEncoder(  # Default parameters for the Clay base model
            img_size=img_size,
            patch_size=8,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
        ).to(device)
        self.img_size = img_size
        self.device = torch.device(device)
        if ckpt_path:
            load_encoder_weights(self.clay_encoder, ckpt_path, device=str(self.device))

    def forward(self, datacube: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.clay_encoder(datacube)

    def fake_datacube(self) -> dict[str, torch.Tensor]:
        "Generate a fake datacube for model export."
        dummy_datacube = {
            "pixels": torch.randn(2, 3, self.img_size, self.img_size),
            "time": torch.randn(2, 4),
            "latlon": torch.randn(2, 4),
            "waves": torch.randn(3),
            "gsd": torch.randn(1),
        }
        return {k: v.to(self.device) for k, v in dummy_datacube.items()}

    def export_to_onnx(self, name: str) -> Any:
        "Save the model to ONNX format."

        datacube = self.fake_datacube()
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)  # ty: ignore[unresolved-attribute]

        # Export the model to ONNX format
        onnx_program = torch.onnx.dynamo_export(  # ty: ignore[unresolved-attribute]
            self.eval(), datacube, export_options=export_options
        )

        # Save the exported model
        onnx_program.save(f"checkpoints/compiled/{name}")
        print(f"Model exported to ONNX format: checkpoints/compiled/{name}")

        return onnx_program

    def export_to_torchep(self, name: str) -> Any:
        "Save the model to pytorch ExportedProgram format."

        datacube = self.fake_datacube()

        # dynamic shapes for model export
        batch_size = Dim("batch_size", min=2, max=1000)
        channel_bands = Dim("channel_bands", min=1, max=10)
        dynamic_shapes = {
            "datacube": {
                "pixels": {0: batch_size, 1: channel_bands},
                "time": {0: batch_size},
                "latlon": {0: batch_size},
                "waves": {0: channel_bands},
                "gsd": {0: None},
            }
        }

        # Export the model to pytorch ExportedProgram format
        ep = torch.export.export(
            self.eval(),
            (datacube,),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )

        # Save the exported model
        torch.export.save(ep, f"checkpoints/compiled/{name}")
        print(f"Model exported to pytorch ExportedProgram format: checkpoints/compiled/{name}")

        return ep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the Clay model.")
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size for the model",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/clay-v1-base.ckpt",
        help="Path to the Clay model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for the model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="clay-base.pt",
        help="Name of the exported model",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Export the model to ONNX format",
    )
    parser.add_argument(
        "--ep",
        action="store_true",
        help="Export the model to pytorch ExportedProgram format",
    )

    args = parser.parse_args()

    Path("checkpoints/compiled").mkdir(parents=True, exist_ok=True)
    embedder = Embedder(
        img_size=args.img_size,
        ckpt_path=args.ckpt_path,
        device=args.device,
    )

    if args.onnx:
        embedder.export_to_onnx(args.name)
    elif args.ep:
        embedder.export_to_torchep(args.name)
    else:
        print("Please specify the format to export the model.")
        parser.print_help()
