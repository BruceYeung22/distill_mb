# AGENTS.md — `src/moebius_finetune/students/`

Student architectures for the GRT hole-fill student. Everything inherits
the conventions in [`../../../AGENTS.md`](../../../AGENTS.md) — the
mid-gray hole-fill, the 9-channel input layout, the GMac / param budget
for RK3588 INT8.

## Files

| File | Public class(es) | Role |
|------|------------------|------|
| `common.py` | `InvertedResidualBlock`, `ContextBlock`, `DownsampleBlock`, `UpsampleBlock`, `TailRefineBlock`, `fuse_bn` | Shared building blocks (MBConv-style + BN folding for ONNX export) |
| `latent.py` | `LatentStudentV0`, `LightweightCodec`, `OneStepLatentNet` | Earlier one-step latent student (legacy; superseded by `mobile_moebius`/`moebius_small`) |
| `mobile_moebius.py` | `MobileMoebius` | 10-step schedule student, 10 teacher forward/trajectory; trajectory-cache distillation target |
| `pixel.py` | `PixelStudentV0` | Pixel-domain student (5-channel cond + noise injection) |
| `schedule.py` | — | DDIM schedule utilities shared with teacher wrapper |
| `gated.py` | `GatedDW7Block` | `x + pw2(a * silu(dw(b)))`, BN + SiLU gate; `expand=1`; NPU-friendly |
| `moebius_small.py` | `MoebiusSmallStudent`, `StudentOutput`, `SinusoidalTimeEmbedding` | **Active target**: ~10.2M params / ~6.4 GMac at `[1,9,64,64]` |

## `MoebiusSmallStudent` (active target)

**Not re-exported by `students/__init__.py`.** Import directly:

```python
from moebius_finetune.students.moebius_small import (
    MoebiusSmallStudent, StudentOutput, SinusoidalTimeEmbedding,
)
from moebius_finetune.students.gated import GatedDW7Block
```

### Signature (locked)

```
MoebiusSmallStudent(in_channels=9, out_channels=4,
                    channels=(128, 256, 512), blocks=(5, 5, 5),
                    time_dim=256)
```

- 9-channel input `[x_t(4) | latent_mask(1) | masked_latent(4)]`.
- **Depth is dropped** at the student — `build_case` still returns `depth_hole` but nothing consumes it.
- Three taps from each stage's last block (post-residual): `[64²×128, 32²×256, 16²×512]`.
- Attention (softmax MQA ×2) only at 16²: last of `enc2` + first of `dec2`.
- T = continuous sinusoidal (`get_timestep_embedding`, `flip_sin_to_cos=True`, `freq_shift=0`), injected 3× per stage.
- BN + ReLU6 (gate SiLU).

### Forward

```python
@dataclass
class StudentOutput:
    sample: torch.Tensor          # eps [B, 4, 64, 64]
    block_outputs: list[Tensor]   # [enc0(128@64²), enc1(256@32²), enc2(512@16²)]

def forward(x: Tensor[B,9,64,64], t: Tensor[B]) -> StudentOutput: ...
```

### Constraints

- Param budget **[9.0M, 12.5M]** (measured: 10,158,695).
- Per-block param formula for `GatedDW7Block(c)`: `3c² + 51c` (BN `2c`, pw1 `2c²`, dw `49c`, pw2 `c²`).
- Anchor params for adaptive-weight loss: `student.enc2[-2].pw2.weight` (skip last MQA block), `student.head[-1].weight`. Both must be **leaf** tensors with finite grads.
- **No `nn.Embedding` lookup for `t`** — continuous sinusoidal only.
- **No `torch.scaled_dot_product_attention`** — plain softmax MQA.

## Common gotchas

- **No CFG.** Guidance is baked in; the student is single-pass.
- `block_outputs` are taken **after** residual addition at each stage's last block — order is fixed `[64², 32², 16²]`.
- Wrong `in_channels` → `ValueError`; wrong `t` shape → `ValueError`.
- For `compute_algorithm_macs` the student has two forward inputs `(x, t)`; pass an `adapter`:
  ```python
  budget.compute_algorithm_macs(
      model, (1, 9, 64, 64),
      adapter=lambda m, spec: (torch.zeros(spec), torch.zeros(spec[0], dtype=torch.long)),
  )
  ```
