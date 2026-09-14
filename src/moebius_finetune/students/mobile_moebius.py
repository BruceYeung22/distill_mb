"""MobileMoebius — iterative 10-step diffusion student (TDD3).

Spec: ``tdd/moebius-student-distill-2026-09-14.md`` (workspace root,
not tracked by git). The model distills the depth-conditioned Moebius
teacher (S2v2) into an RK3588-friendly UNet that runs **10 discrete
denoising steps** in latent space.

Design decisions carried over from TDD3:

* **S1** — the whole 16×16 stage operates at 512 channels
  (``channels=(32, 128, 512)``), measured at 8.66 M params / 2.63 GMacs.
* **S2** — ``in_channels=11``: the teacher's 9-channel latent contract
  (``[x_t(4), masked_latent(4), mask(1)]``) plus the §4.3 depth
  features ``[depth_mean(1), coverage(1)]`` computed by
  :func:`moebius_finetune.training.teacher.finetune._default_depth_features`.
  The depth features are hole-zeroed upstream (anti-leakage contract);
  this module consumes them as-is.

Deployment notes (RK3588): all blocks are DW/PW convolutions with
foldable ``BatchNorm2d`` and ``ReLU6``; the bottleneck attention uses
explicit ``matmul → softmax → matmul`` (multi-query, shared K/V head)
instead of ``torch.scaled_dot_product_attention``; upsampling is
nearest-neighbour. The :class:`TimeEmbedding` lookup table over the 10
fixed steps can be precomputed to constants at export time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ConvBNAct",
    "Downsample",
    "MobileBottleneck",
    "MobileFFN",
    "MobileMoebius",
    "MobileMQA",
    "RKBlock",
    "TimeEmbedding",
    "TimeInject",
    "Upsample",
]


# ---------------------------------------------------------------------------
# Basic NPU-friendly blocks
# ---------------------------------------------------------------------------


class ConvBNAct(nn.Sequential):
    """Conv (bias-free) + foldable BatchNorm + optional ReLU6."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 1,
        s: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        p = k // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU6(inplace=False))
        super().__init__(*layers)


class RKBlock(nn.Module):
    """PW → DW3×3 → PW residual block (TDD3 §1, RK3588-friendly).

    ``x: [B, C, H, W]`` → ``x + block(x)`` with the projection left
    un-activated so the residual add is numerically well-behaved and
    the BN folds cleanly at export time.
    """

    def __init__(self, channels: int, expand: int = 2) -> None:
        super().__init__()
        hidden = int(channels * expand)
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden, k=1, act=True),
            ConvBNAct(hidden, hidden, k=3, groups=hidden, act=True),
            ConvBNAct(hidden, channels, k=1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Downsample(nn.Module):
    """DW stride-2 → PW."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, in_ch, k=3, s=2, groups=in_ch, act=True),
            ConvBNAct(in_ch, out_ch, k=1, act=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Upsample(nn.Module):
    """Nearest ×2 → DW → PW."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=3, groups=in_ch, act=True)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# Timestep conditioning
# ---------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    """Embedding over the fixed 10-step schedule + SiLU MLP.

    ``step_id ∈ {0 .. num_steps-1}`` indexes the discrete DDIM
    sub-schedule (TDD3 §2). Deployment may replace the table with
    precomputed constant vectors.
    """

    def __init__(self, num_steps: int = 10, dim: int = 256) -> None:
        super().__init__()
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self.num_steps = int(num_steps)
        self.embedding = nn.Embedding(self.num_steps, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, step_id: torch.Tensor) -> torch.Tensor:
        step_id = torch.as_tensor(step_id, device=self.embedding.weight.device)
        if step_id.dtype != torch.long:
            step_id = step_id.long()
        lo = int(step_id.min().item())
        hi = int(step_id.max().item())
        if lo < 0 or hi >= self.num_steps:
            raise ValueError(
                f"step_id out of range for a {self.num_steps}-step schedule: "
                f"got min={lo}, max={hi}, expected values in [0, {self.num_steps - 1}]"
            )
        return self.mlp(self.embedding(step_id))


class TimeInject(nn.Module):
    """Project the time embedding to per-channel biases: ``[B, C, 1, 1]``."""

    def __init__(self, time_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.proj(t)[:, :, None, None]
        return x + t


# ---------------------------------------------------------------------------
# Mobile multi-query attention
# ---------------------------------------------------------------------------


class MobileMQA(nn.Module):
    """Multi-query attention with shared K/V (RKNN-friendly op set).

    Q: ``q_heads × head_dim``; K/V: one shared head of ``head_dim``.
    Implemented as explicit ``matmul → softmax → matmul`` on purpose —
    no ``torch.scaled_dot_product_attention`` — so the graph maps 1:1
    onto RKNN operator kernels.
    """

    def __init__(self, channels: int = 256, q_heads: int = 4, head_dim: int = 32) -> None:
        super().__init__()
        if channels % q_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by q_heads ({q_heads})"
            )
        self.channels = channels
        self.q_heads = q_heads
        self.head_dim = head_dim
        q_dim = q_heads * head_dim

        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, q_dim, bias=False)
        self.k_proj = nn.Linear(channels, head_dim, bias=False)
        self.v_proj = nn.Linear(channels, head_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, channels, bias=False)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` → ``[B, C, H, W]`` (residual attention)."""
        residual = x
        B, C, H, W = x.shape
        N = H * W

        # BCHW → BNC
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        # Q: [B, Hq, N, D]
        q = self.q_proj(x).view(B, N, self.q_heads, self.head_dim).permute(0, 2, 1, 3)
        # shared K / V: [B, 1, N, D]
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # [B, Hq, N, D] → [B, N, C] → BCHW
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, N, self.q_heads * self.head_dim)
        out = self.out_proj(out)
        return residual + out.transpose(1, 2).reshape(B, C, H, W)


class MobileFFN(nn.Module):
    """Mobile FFN: PW → DW3×3 → PW with a residual add."""

    def __init__(self, channels: int = 256, expand: int = 2) -> None:
        super().__init__()
        hidden = int(channels * expand)
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden, k=1, act=True),
            ConvBNAct(hidden, hidden, k=3, groups=hidden, act=True),
            ConvBNAct(hidden, channels, k=1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MobileBottleneck(nn.Module):
    """MQA → FFN → MQA at the lowest resolution (TDD3 §1)."""

    def __init__(
        self,
        channels: int = 256,
        q_heads: int = 4,
        head_dim: int = 32,
        ffn_expand: int = 2,
    ) -> None:
        super().__init__()
        self.mqa1 = MobileMQA(channels, q_heads, head_dim)
        self.ffn = MobileFFN(channels, expand=ffn_expand)
        self.mqa2 = MobileMQA(channels, q_heads, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mqa2(self.ffn(self.mqa1(x)))


# ---------------------------------------------------------------------------
# MobileMoebius
# ---------------------------------------------------------------------------


class MobileMoebius(nn.Module):
    """10-step latent diffusion student distilled from the S2v2 teacher.

    Inputs
    ------
    x:
        ``[B, in_channels, 64, 64]`` — the teacher's 9-channel latent
        contract plus the two §4.3 depth features by default
        (``[x_t, masked_latent, mask, depth_mean, coverage]``).
    step_id:
        ``[B]`` int64 schedule index in ``[0, num_steps)``.

    Output
    ------
    ``[B, out_channels, 64, 64]`` — predicted noise (epsilon), same
    convention as the teacher.
    """

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 4,
        channels: tuple[int, int, int] = (32, 128, 512),
        blocks: tuple[int, int, int, int, int, int] = (2, 3, 3, 3, 3, 2),
        num_steps: int = 10,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive, got "
                f"{(in_channels, out_channels)}"
            )
        if len(channels) != 3 or any(c <= 0 for c in channels):
            raise ValueError(f"channels must be 3 positive ints, got {channels}")
        if len(blocks) != 6 or any(b < 1 for b in blocks):
            raise ValueError(f"blocks must be 6 ints >= 1, got {blocks}")
        c0, c1, c2 = channels

        # ----------------------- timestep -----------------------------
        self.time_embed = TimeEmbedding(num_steps=num_steps, dim=time_dim)
        self.t0 = TimeInject(time_dim, c0)
        self.t1 = TimeInject(time_dim, c1)
        self.t2 = TimeInject(time_dim, c2)

        # ----------------------- conv_in ------------------------------
        self.conv_in = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3, groups=in_channels, act=True),
            ConvBNAct(in_channels, c0, k=1, act=False),
        )

        # ----------------------- encoder ------------------------------
        self.enc0 = nn.Sequential(*[RKBlock(c0) for _ in range(blocks[0])])
        self.down0 = Downsample(c0, c1)
        self.enc1 = nn.Sequential(*[RKBlock(c1) for _ in range(blocks[1])])
        self.down1 = Downsample(c1, c2)
        self.enc2 = nn.Sequential(*[RKBlock(c2) for _ in range(blocks[2])])

        # ----------------------- bottleneck ---------------------------
        self.bottleneck = MobileBottleneck(channels=c2, q_heads=4, head_dim=32)

        # ----------------------- decoder ------------------------------
        self.dec2 = nn.Sequential(*[RKBlock(c2) for _ in range(blocks[3])])
        self.up1 = Upsample(c2, c1)
        self.dec1 = nn.Sequential(*[RKBlock(c1) for _ in range(blocks[4])])
        self.up0 = Upsample(c1, c0)
        self.dec0 = nn.Sequential(*[RKBlock(c0) for _ in range(blocks[5])])

        # ----------------------- head ---------------------------------
        self.head = nn.Sequential(
            ConvBNAct(c0, c0, k=3, groups=c0, act=True),
            nn.Conv2d(c0, out_channels, kernel_size=1),
        )
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, step_id: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        temb = self.time_embed(step_id)

        # encoder 64²
        x = self.conv_in(x)
        x = self.t0(x, temb)
        x = self.enc0(x)
        skip0 = x

        # encoder 32²
        x = self.down0(x)
        x = self.t1(x, temb)
        x = self.enc1(x)
        skip1 = x

        # encoder 16²
        x = self.down1(x)
        x = self.t2(x, temb)
        x = self.enc2(x)

        # bottleneck 16²
        x = self.bottleneck(x)

        # decoder 16²
        x = self.dec2(x)

        # decoder 32²
        x = self.up1(x)
        x = x + skip1
        x = self.t1(x, temb)
        x = self.dec1(x)

        # decoder 64²
        x = self.up0(x)
        x = x + skip0
        x = self.t0(x, temb)
        x = self.dec0(x)

        return self.head(x)
