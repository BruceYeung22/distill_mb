import pytest


def test_epsilon_mse_mixed_empty_cases_uses_full_batch_mean_and_gradient():
    torch = pytest.importorskip("torch")
    from moebius_finetune.training.teacher.losses import epsilon_mse

    pred = torch.ones(3, 4, 2, 2, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.ones(3, 1, 2, 2)
    mask[1] = 0
    out = epsilon_mse(pred, target, mask)
    assert torch.allclose(out, torch.tensor(2.0 / 3.0))
    out.backward()
    assert pred.grad[1].abs().sum() == 0
    assert torch.all(pred.grad[[0, 2]] > 0)
