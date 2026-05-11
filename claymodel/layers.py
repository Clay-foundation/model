"""Transformer code adapted from Phil Wang's vit-pytorch library.
Repository: https://github.com/lucidrains/vit-pytorch
"""

__all__ = ["Attention", "FeedForward", "Transformer"]

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    heads: int
    scale: float
    fused_attn: bool

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, fused_attn: bool = True
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.fused_attn = fused_attn

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)

        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        fused_attn: bool,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, fused_attn=fused_attn),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            attn, ff = layer[0], layer[1]  # ty: ignore[not-subscriptable]
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
