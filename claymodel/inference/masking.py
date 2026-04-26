"""Patch-level analysis utilities for Clay model inputs."""

import torch
from einops import rearrange


class PatchAnalyzer:
    """Analyze input chips at the patch level for quality filtering.

    Computes per-patch metrics like valid pixel fraction and cloud fraction
    that can be used to filter out low-quality chips before embedding.

    Note: Clouds produce valid embeddings in Clay by design (the model was
    trained WITH clouds). Use cloud_fraction for downstream filtering if
    your application requires cloud-free data, not to pre-filter model input.

    Args:
        patch_size: Patch size matching the model config (default 8).
        nodata: Value indicating missing/nodata pixels. Can be a scalar
            or "nan" to detect NaN values. Default is 0.
    """

    patch_size: int
    nodata: float | str

    def __init__(self, patch_size: int = 8, nodata: float | str = 0) -> None:
        self.patch_size = patch_size
        self.nodata = nodata

    def valid_fraction(self, pixels: torch.Tensor) -> torch.Tensor:
        """Fraction of valid (non-nodata) pixels per patch.

        Should be called on RAW (pre-normalization) pixel values, since
        normalization transforms nodata values into non-zero values.

        Args:
            pixels: [B, C, H, W] raw pixel values (before normalization).

        Returns:
            [B, num_patches] tensor of valid fractions (0.0 to 1.0).
        """
        if self.nodata == "nan":
            is_valid = ~torch.isnan(pixels).any(dim=1)  # [B, H, W]
        else:
            is_valid = (pixels != self.nodata).any(dim=1)  # [B, H, W]

        patches = rearrange(
            is_valid.float(),
            "B (h p1) (w p2) -> B (h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return patches.mean(dim=-1)  # [B, num_patches]

    def cloud_fraction(self, scl_band: torch.Tensor) -> torch.Tensor:
        """Fraction of cloudy pixels per patch using Sentinel-2 SCL band.

        Uses the Scene Classification Layer (SCL) from Sentinel-2 L2A data.
        SCL cloud classes: 8 (medium probability), 9 (high probability),
        10 (thin cirrus).

        Note: This is Sentinel-2 specific. For other sensors, cloud masking
        must be handled externally.

        Args:
            scl_band: [B, H, W] Scene Classification Layer integer values.

        Returns:
            [B, num_patches] tensor of cloud fractions (0.0 to 1.0).
        """
        is_cloud = (scl_band == 8) | (scl_band == 9) | (scl_band == 10)
        patches = rearrange(
            is_cloud.float(),
            "B (h p1) (w p2) -> B (h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return patches.mean(dim=-1)  # [B, num_patches]

    def filter_chips(
        self,
        pixels: torch.Tensor,
        min_valid: float = 0.5,
        scl_band: torch.Tensor | None = None,
        max_cloud: float = 1.0,
    ) -> torch.Tensor:
        """Filter chips by valid fraction and optionally cloud fraction.

        Args:
            pixels: [B, C, H, W] raw pixel values.
            min_valid: Minimum fraction of valid pixels to keep chip.
            scl_band: Optional [B, H, W] SCL band for cloud filtering.
            max_cloud: Maximum cloud fraction to keep chip (requires scl_band).

        Returns:
            Boolean mask [B] where True means the chip passes filtering.
        """
        vf = self.valid_fraction(pixels)
        # A chip passes if its minimum patch valid fraction >= min_valid
        chip_valid = vf.min(dim=-1).values >= min_valid

        if scl_band is not None and max_cloud < 1.0:
            cf = self.cloud_fraction(scl_band)
            chip_cloud = cf.max(dim=-1).values <= max_cloud
            return chip_valid & chip_cloud

        return chip_valid
