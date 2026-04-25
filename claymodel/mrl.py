"""
Matryoshka Representation Learning (MRL).

MRL was used for approximately 90% of the Clay v1.5 training run (~epochs 1-70)
before being replaced with a direct linear projection (nn.Linear) for the
remaining training. This module is retained because:

1. The trained checkpoint may contain MRL-related weight keys that need to be
   handled (ignored) during loading.
2. It documents the training history of v1.5.

The active model (ClayMAE in model.py) uses self.proj instead of self.mrl.
See model.py:393 for the comment explaining the transition.
"""

from torch import nn


class MRL(nn.Module):
    """
    Matryoshka Representation Learning from the paper: https://arxiv.org/abs/2205.13147
    """

    def __init__(self, features, dolls: list = [16, 32, 64, 128, 256, 768]) -> None:
        super().__init__()
        self.dolls = dolls
        self.layers = nn.ModuleDict()
        for doll in dolls:
            self.layers[f"mrl_{doll}"] = nn.Linear(doll, features)

    def forward(self, x):
        "x: (batch, features)"
        logits = [self.layers[f"mrl_{doll}"](x[:, :doll]) for doll in self.dolls]
        return logits


class MRLLoss(nn.Module):
    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights
        self.criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, representations, targets):
        """
        representations: [(batch, features), ...]
        targets: (batch, features)
        """
        losses = [
            self.weights[i] * (1 - self.criterion(rep, targets)).mean()
            for i, rep in enumerate(representations)
        ]
        return sum(losses) / len(losses)
