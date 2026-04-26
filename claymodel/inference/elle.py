"""
ELLE: Embedding-Level Loss Estimation.

A lightweight Ridge regression probe that predicts reconstruction difficulty
from the CLS embedding at ~1us/sample (no decoder needed).

The probe is trained by fitting sklearn Ridge regression on:
  X = CLS embeddings from encoder [N, D]
  y = per-sample reconstruction loss from full MAE forward pass [N]

The quality score correlates with actual reconstruction loss (>0.9 correlation)
and indicates how "hard" an image is for the model to reconstruct. Higher scores
indicate more complex or unusual inputs.

Reference: https://devlogs.lgnd.ai/posts/2026-03-01-self-aware-embeddings/
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import torch


class ELLEProbe:
    """Predict reconstruction difficulty from CLS embedding.

    The probe is a simple linear model (Ridge regression coefficients)
    that maps CLS embeddings to predicted reconstruction loss.

    Args:
        coef: [D] tensor of regression coefficients.
        intercept: Scalar bias term.
    """

    def __init__(self, coef: torch.Tensor, intercept: float):
        self.coef = coef
        self.intercept = intercept

    @classmethod
    def from_file(cls, path: str | Path) -> ELLEProbe:
        """Load a pre-trained ELLE probe from a .pt file.

        Args:
            path: Path to a .pt file containing {"coef": tensor, "intercept": float}.

        Returns:
            ELLEProbe instance.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(coef=data["coef"], intercept=float(data["intercept"]))

    @classmethod
    def default(cls) -> ELLEProbe:
        """Load the bundled pre-trained ELLE probe.

        Raises:
            FileNotFoundError: If the bundled probe weights are not available.
        """
        probe_path = files("claymodel").joinpath("configs/elle_probe.pt")
        if not Path(str(probe_path)).exists():
            raise FileNotFoundError(
                "Bundled ELLE probe weights not found. "
                "Train a probe using scripts/train_elle_probe.py "
                "or provide weights via ELLEProbe.from_file()."
            )
        return cls.from_file(str(probe_path))

    def score(self, cls_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict reconstruction difficulty from CLS embeddings.

        Args:
            cls_embeddings: [N, D] tensor of CLS token embeddings from encoder.

        Returns:
            [N] tensor of quality scores (predicted reconstruction loss).
            Higher values indicate harder-to-reconstruct inputs.
        """
        coef = self.coef.to(cls_embeddings.device)
        return cls_embeddings @ coef + self.intercept

    def save(self, path: str | Path):
        """Save probe weights to a .pt file."""
        torch.save(
            {"coef": self.coef.cpu(), "intercept": self.intercept},
            path,
        )
