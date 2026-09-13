"""Small regression tests for teacher microbatch accumulation semantics."""

from __future__ import annotations

import pytest


def _batch(torch, value: float):
    shape = (1, 1, 16, 16)
    latent_shape = (1, 4, 2, 2)
    return {
        "clean_rgb": torch.full((1, 3, 16, 16), value),
        "hole_mask": torch.zeros(shape),
        "depth_hole": torch.zeros(shape),
        "noise": torch.full(latent_shape, value),
        "masked_latent": torch.full(latent_shape, value),
        "clean_latent": torch.full(latent_shape, value),
    }


def _run_microbatch_series(torch, finetune, model, optimizer, batches, seed):
    torch.manual_seed(seed)
    optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        finetune._train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            vae=None,
            device=torch.device("cpu"),
            amp=False,
            amp_dtype=None,
            known_weight=0.1,
            alphas_bar=torch.ones(1000),
            loss_scale=0.5,
        )
    return [p.grad.detach().clone() for p in model.parameters()]


def test_train_step_accumulates_scaled_gradients(torch):
    from moebius_finetune.training.teacher import finetune

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(9, 4, kernel_size=1, bias=False)

        def forward(self, x, timesteps, input_ids, *, depth_features=None):
            return self.proj(x)

    torch.manual_seed(11)
    initial = Tiny().state_dict()
    batches = [_batch(torch, 0.2), _batch(torch, 0.7)]

    accumulated = Tiny()
    accumulated.load_state_dict(initial)
    accumulated_opt = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    actual = _run_microbatch_series(
        torch, finetune, accumulated, accumulated_opt, batches, seed=23
    )

    reference = Tiny()
    reference.load_state_dict(initial)
    reference_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    torch.manual_seed(23)
    reference_grads = []
    for batch in batches:
        reference_opt.zero_grad(set_to_none=True)
        finetune._train_step(
            model=reference,
            optimizer=reference_opt,
            batch=batch,
            vae=None,
            device=torch.device("cpu"),
            amp=False,
            amp_dtype=None,
            known_weight=0.1,
            alphas_bar=torch.ones(1000),
            loss_scale=1.0,
        )
        reference_grads.append(next(reference.parameters()).grad.detach().clone())
    expected = [(reference_grads[0] + reference_grads[1]) / 2.0]

    assert torch.allclose(actual[0], expected[0], atol=1e-6, rtol=1e-5)
    accumulated_opt.step()
    expected_param = initial["proj.weight"] - 0.1 * expected[0]
    assert torch.allclose(
        accumulated.proj.weight, expected_param, atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("grad_accum_steps", [1, 2])
def test_loop_counts_optimizer_updates_and_microbatches(
    torch, tmp_path, monkeypatch, grad_accum_steps
):
    from moebius_finetune.teachers.depth_adapter import DepthConditionAdapter
    from moebius_finetune.teachers.wrapper import DepthConditionedRemoval
    from moebius_finetune.training.teacher import finetune
    from moebius_finetune.training.teacher.recipe import DepthBranchOnlyRecipe

    class TinyBackbone(torch.nn.Module):
        num_embeddings = 10

        def __init__(self):
            super().__init__()
            self.diff_model = torch.nn.Module()
            class ConvIn(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv_pw = torch.nn.Conv2d(9, 4, kernel_size=1)

                def forward(self, x):
                    return self.conv_pw(x)

            self.diff_model.conv_in = ConvIn()

        def forward(self, x, timesteps, input_ids):
            return self.diff_model.conv_in(x)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = DepthConditionedRemoval(
        TinyBackbone(), DepthConditionAdapter(hidden=2, target_dim=4)
    )
    cfg = DepthBranchOnlyRecipe(
        steps=3,
        lr=1e-2,
        grad_accum_steps=grad_accum_steps,
        log_every_steps=1,
        save_every_steps=1,
        amp=False,
    )
    calls = {"provider": 0, "clip": 0}

    def provider():
        calls["provider"] += 1
        return _batch(torch, 0.1 * calls["provider"])

    original_clip = finetune.nn.utils.clip_grad_norm_

    def clip(*args, **kwargs):
        calls["clip"] += 1
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(finetune.nn.utils, "clip_grad_norm_", clip)
    artifacts = finetune.finetune_depth_branch(
        model,
        cfg,
        batch_provider=provider,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert calls == {"provider": 3 * grad_accum_steps, "clip": 3}
    assert artifacts.global_step == 3
    assert len(artifacts.checkpoint_paths) == 3  # one checkpoint per update
    assert (tmp_path / "teacher_step00000000.pt").is_file()
    assert [p.name for p in artifacts.checkpoint_paths] == [
        "teacher_step00000001.pt",
        "teacher_step00000002.pt",
        "teacher_step00000003.pt",
    ]
    log = artifacts.log_path.read_text(encoding="utf-8")
    assert len([line for line in log.splitlines() if line.startswith("step=")]) == 3
    assert "step=1 " in log and "step=2 " in log and "step=3 " in log
    payload = torch.load(artifacts.checkpoint_paths[-1], map_location="cpu", weights_only=False)
    steps = [state["step"] for state in payload["optimizer_state"]["state"].values()]
    assert steps and all(int(step) == 3 for step in steps)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")
