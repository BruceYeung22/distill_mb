"""Behavioral regressions for executed budgets."""
import pytest
import torch
from torch import nn
from moebius_finetune.deployment.budget import compute_algorithm_macs, compute_logical_traffic, compute_combined_macs, compare_to_budget, estimate_dram_traffic
from moebius_finetune.students.common import InvertedResidualBlock


def test_batched_rectangular_convolution_and_transpose():
    conv = nn.Conv2d(4, 6, (3, 5), groups=2, bias=False)
    assert compute_algorithm_macs(conv, (2, 4, 8, 10))["total_macs"] == 2*6*6*6*2*3*5
    trans = nn.ConvTranspose2d(4, 6, (3, 5), stride=2, groups=2, bias=False)
    assert compute_algorithm_macs(trans, (2, 4, 8, 10))["total_macs"] == 2*4*8*10*3*3*5


def test_every_conv_reads_its_input():
    model = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), nn.Conv2d(4, 4, 1, bias=False))
    assert compute_logical_traffic(model, (1, 4, 8, 8))["total_bytes"] == 32 + 4*256


def test_resize_is_recorded_once_and_functional_add_is_counted():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2)
        def forward(self, x):
            y = self.up(x)
            return y + y
    r = compute_algorithm_macs(Model(), (1, 4, 8, 8))
    assert sum(op["name"].startswith("Resize") for op in r["operators"]) == 1
    assert r["total_macs"] == 4*16*16
    assert r["accounting_complete"]


def test_unknown_cost_cannot_pass_either_budget_axis():
    class Unknown(nn.Module):
        def forward(self, x):
            return torch.sin(x)
    r = compute_algorithm_macs(Unknown(), (1, 4, 8, 8))
    assert not r["accounting_complete"]
    assert r["unknown_ops"]
    combined = compute_combined_macs([(Unknown(), (1, 4, 8, 8))])
    assert not combined["accounting_complete"]
    comp = compare_to_budget({**r, "dram_traffic": estimate_dram_traffic(0)})
    assert comp["algorithm_budget_pass"] is False
    assert comp["traffic_estimate_pass"] is False


def test_fusion_preserves_float64_and_outputs():
    model = InvertedResidualBlock(4).double().eval()
    x = torch.randn(2, 4, 8, 8, dtype=torch.float64)
    fused = model.fuse().eval()
    assert all(p.dtype == torch.float64 for p in fused.parameters())
    torch.testing.assert_close(fused(x), model(x))


def test_shared_conv_counts_each_execution_but_unique_parameters():
    conv = nn.Conv2d(4, 4, 1, bias=False)
    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = conv
        def forward(self, x):
            return self.conv(self.conv(x))
    r = compute_algorithm_macs(Twice(), (1, 4, 8, 8))
    assert r["total_macs"] == 2*4*4*8*8
    assert r["total_weight_params"] == 16
    assert compute_logical_traffic(Twice(), (1, 4, 8, 8))["total_bytes"] == 32 + 4*256
    assert compute_combined_macs([(conv, (1,4,8,8)), (conv, (1,4,8,8))])["total_weight_params"] == 16


@pytest.mark.parametrize("kind", ["pixel", "codec", "latent"])
def test_student_conv_counts_match_executed_hook_oracle(kind):
    from moebius_finetune.students import PixelStudentV0, LatentStudentV0
    from moebius_finetune.deployment.budget import _default_inputs
    model = PixelStudentV0() if kind == "pixel" else LatentStudentV0()
    if kind == "codec":
        model = model.codec
    model.eval()
    spec = (1, 3 if kind == "codec" else 5, 64, 64)
    actual = []
    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(
                lambda m, args, out: actual.append(out.numel()*m.weight[0].numel())))
    with torch.no_grad():
        model(*_default_inputs(model, spec))
    for handle in handles:
        handle.remove()
    r = compute_algorithm_macs(model, spec)
    assert sum(op["macs"] for op in r["operators"] if op["name"] in {"Conv", "DepthwiseConv"}) == sum(actual)
    expected_shape = [1, 4, 8, 8] if kind == "codec" else [1, 3, 64, 64]
    assert r["output_shape"] == expected_shape
    if kind != "latent":
        assert r["accounting_complete"], r["unknown_ops"]
    else:
        assert any("NumPy" in op for op in r["unknown_ops"])
