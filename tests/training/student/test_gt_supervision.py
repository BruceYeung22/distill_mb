"""A real-label perturbation must reach each student's parameter update."""

import copy

import numpy as np
import pytest


@pytest.mark.parametrize("kind", ["pixel", "latent"])
def test_changing_only_gt_changes_student_update(kind):
    torch = pytest.importorskip("torch")
    from moebius_finetune.students import LatentStudentV0, PixelStudentV0
    from moebius_finetune.training.student import SyntheticTeacherCache, distill_student

    torch.manual_seed(37)
    initial = (PixelStudentV0 if kind == "pixel" else LatentStudentV0)()
    entry = SyntheticTeacherCache(resolution=64, num_cases=1, seed=11).entries()[0]
    updated = []
    for gt_value in (0.0, 1.0, None):
        student = copy.deepcopy(initial)
        case = copy.deepcopy(entry)
        case.target_rgb = (
            np.full_like(case.teacher_rgb, gt_value) if gt_value is not None else None
        )
        trainer = distill_student(
            student, [case],
            cfg={"steps": 1, "lr": 1e-3, "weight_decay": 0.0, "device": "cpu"},
        )
        if gt_value is None:
            assert trainer.history[0]["loss_rgb_hole_gt"] == 0.0
        else:
            updated.append([p.detach().clone() for p in student.parameters()])
    assert any(not torch.equal(a, b) for a, b in zip(*updated)), (
        "Changing only the ground truth did not change any learned parameter"
    )
