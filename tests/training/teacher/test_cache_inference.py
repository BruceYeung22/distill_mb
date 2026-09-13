"""Regression coverage for the teacher inference boundary."""

import numpy as np
import pytest


def _tiny_components():
    torch = pytest.importorskip("torch")
    from torch import nn
    from moebius_finetune.contracts import ConditionBatch

    class Base(nn.Module):
        num_embeddings = 10

        def __init__(self):
            super().__init__()
            self.diff_model = nn.Module()
            class ConvIn(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv_pw = nn.Conv2d(9, 4, 1)
                def forward(self, x):
                    return self.conv_pw(x)
            self.diff_model.conv_in = ConvIn()
            self.bn = nn.BatchNorm2d(4)
            self.fail = False

        def forward(self, x, t, ids):
            if self.fail:
                raise RuntimeError("synthetic forward failure")
            return self.bn(self.diff_model.conv_in(x))

    class VAE(nn.Module):
        class config:
            scaling_factor = 1.0

        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(()))

        def encode(self, x):
            z = torch.nn.functional.avg_pool2d(x, 8).mean(1, keepdim=True).expand(-1, 4, -1, -1)
            return type("Encoded", (), {"latent_dist": type("Dist", (), {"mode": lambda s: z})()})()

        def decode(self, z):
            return type("Decoded", (), {"sample": torch.nn.functional.interpolate(z[:, :3], scale_factor=8)})()

    H = W = 16
    mask = np.ones((1, 1, H, W), np.float32)
    rgb = np.zeros((1, 3, H, W), np.float32)
    depth = np.zeros((1, 1, H, W), np.float32)
    noise = np.zeros((1, 4, H // 8, W // 8), np.float32)
    condition = ConditionBatch.from_arrays(rgb, mask, depth, noise)
    return torch, Base(), VAE(), condition


def test_baseline_cache_accepts_depth_features_and_returns_numpy():
    torch, base, vae, condition = _tiny_components()
    from moebius_finetune.teachers.original_baseline import OriginalRemovalBaseline
    from moebius_finetune.training.teacher.cache import cache_teacher_outputs, default_scheduler_config

    cfg = default_scheduler_config(num_inference_steps=2)
    entries = cache_teacher_outputs(
        OriginalRemovalBaseline(base), ["tiny"], [0], cfg, vae=vae,
        case_batch_builder=lambda *_: (condition, torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 2, 2)),
    )
    assert isinstance(entries[0].final_latent, np.ndarray)
    assert isinstance(entries[0].teacher_rgb, np.ndarray)


def test_cache_eval_is_no_grad_preserves_mixed_modes_and_bn():
    torch, base, vae, condition = _tiny_components()
    from moebius_finetune.teachers.depth_adapter import DepthConditionAdapter
    from moebius_finetune.teachers.wrapper import DepthConditionedRemoval
    from moebius_finetune.training.teacher.cache import cache_teacher_outputs, default_scheduler_config

    wrapped = DepthConditionedRemoval(base, DepthConditionAdapter(hidden=2, target_dim=4))
    wrapped.model.eval()
    wrapped.depth_adapter.train()
    modules = list(wrapped.modules()) + list(vae.modules())
    modes = {m: m.training for m in modules}
    buffers = {id(m): {k: v.clone() for k, v in m.named_buffers()}
               for m in modules if isinstance(m, torch.nn.BatchNorm2d)}
    grad_modes = []
    hook = base.diff_model.conv_in.register_forward_pre_hook(
        lambda *_: grad_modes.append(torch.is_grad_enabled()))
    cache_teacher_outputs(
        wrapped, ["tiny"], [0], default_scheduler_config(num_inference_steps=2), vae=vae,
        case_batch_builder=lambda *_: (condition, torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 2, 2)),
    )
    assert wrapped.model.training is False
    assert wrapped.depth_adapter.training is True
    assert grad_modes and not any(grad_modes)
    assert all(m.training == mode for m, mode in modes.items())
    for m in modules:
        for key, expected in buffers.get(id(m), {}).items():
            torch.testing.assert_close(dict(m.named_buffers())[key], expected)

    base.fail = True
    with pytest.raises(RuntimeError):
        cache_teacher_outputs(
            wrapped, ["tiny"], [0], default_scheduler_config(num_inference_steps=2), vae=vae,
            case_batch_builder=lambda *_: (condition, torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 2, 2)),
        )
    assert wrapped.model.training is False
    assert wrapped.depth_adapter.training is True
    assert all(m.training == mode for m, mode in modes.items())
    hook.remove()


def test_nodepth_evaluation_loads_stage2_conv_in_before_disabling_adapter(tmp_path, monkeypatch):
    torch, base, _vae, _condition = _tiny_components()
    from moebius_finetune.teachers.depth_adapter import DepthConditionAdapter
    from moebius_finetune.teachers.wrapper import DepthConditionedRemoval
    import moebius_finetune.teachers.loader as loader
    import moebius_finetune.teachers as teachers
    from moebius_finetune.evaluation.teacher_eval import _build_model_for_run

    wrapped = DepthConditionedRemoval(base, DepthConditionAdapter(hidden=2, target_dim=4))
    expected = torch.full_like(wrapped.model.diff_model.conv_in.conv_pw.weight, 3.0)
    state = {"model.diff_model.conv_in.conv_pw.weight": expected}
    state["model.diff_model.conv_in.conv_pw.bias"] = wrapped.model.diff_model.conv_in.conv_pw.bias.detach().clone()
    ckpt = tmp_path / "stage2.pt"
    torch.save({"model_state": state}, ckpt)
    monkeypatch.setattr(loader, "load_removal_model", lambda *_args, **_kwargs: _tiny_components()[1])
    monkeypatch.setattr(teachers, "DepthConditionAdapter", lambda: DepthConditionAdapter(hidden=2, target_dim=4))
    result = _build_model_for_run({"use_depth": False, "checkpoint": str(ckpt)}, "ignored", "cpu")
    torch.testing.assert_close(result.model.diff_model.conv_in.conv_pw.weight, expected)
    assert result._branch_depth_enabled is False
    assert result.depth_adapter.enabled is False
