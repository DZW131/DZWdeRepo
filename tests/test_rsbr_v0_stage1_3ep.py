import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from network.resnet38_cls_rsbr import Net
from tools.rsbr_stage1_contract import (
    GO,
    NOGO,
    REVIEW,
    STRONG_GO,
    decide_pilot,
    select_best_epoch,
)
from tools.run_rsbr_v0_stage1_3ep import (
    EPOCHS,
    build_optimizer,
    frozen_mode_ok,
    set_frozen_training_mode,
)


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "artifacts" / "rsbr_v0_parity_r1_and_readiness_7cbe5aa"
FROZEN_HASHES = {
    "network/rsbr_v0.py": "b13ff51e0b73816fa3ffbf241764f2f50bfcda5d2de39951f165cf86a2e0a80a",
    "network/resnet38_cls_rsbr.py": "6af680e5be3b509ed4ef87d48e118e050fa1445b6a87234c657f88fb3ddf2765",
}


def decision_kwargs():
    return {
        "nonnegative_classes_at_best": 3,
        "positive_classes_at_best": 3,
        "safety_failure": False,
        "mechanism_review_evidence": False,
    }


def test_prerequisite_artifacts_are_double_pass():
    parity = json.loads((PRIOR / "parity_r1" / "summary.json").read_text())
    readiness = json.loads((PRIOR / "readiness_32b" / "summary.json").read_text())
    assert parity["decision"] == "RSBR_V0_PARITY_R1_PASS"
    assert readiness["decision"] == "RSBR_V0_READINESS_PASS"


def test_decision_boundaries_and_safety_precedence():
    assert decide_pilot(
        best_delta_miou_pp=0.30,
        epoch3_delta_miou_pp=0.20,
        **decision_kwargs(),
    ) == STRONG_GO
    assert decide_pilot(
        best_delta_miou_pp=0.15,
        epoch3_delta_miou_pp=0.0,
        **decision_kwargs(),
    ) == GO
    assert decide_pilot(
        best_delta_miou_pp=0.05,
        epoch3_delta_miou_pp=0.0,
        **decision_kwargs(),
    ) == REVIEW
    values = decision_kwargs()
    values["safety_failure"] = True
    assert decide_pilot(
        best_delta_miou_pp=1.0,
        epoch3_delta_miou_pp=1.0,
        **values,
    ) == NOGO


def test_best_epoch_uses_earliest_tie_break():
    rows = [
        {"epoch": 1, "paired_delta_miou_pp": 0.2},
        {"epoch": 2, "paired_delta_miou_pp": 0.2},
        {"epoch": 3, "paired_delta_miou_pp": 0.1},
    ]
    assert select_best_epoch(rows)["epoch"] == 1


def test_model_sources_remain_frozen():
    for relative, expected in FROZEN_HASHES.items():
        canonical = (ROOT / relative).read_text(encoding="utf-8").encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == expected


def test_frozen_mode_and_optimizer_coverage():
    model = Net(n_class=4)
    set_frozen_training_mode(model)
    assert all(frozen_mode_ok(model).values())
    optimizer = build_optimizer(
        model,
        SimpleNamespace(lr=0.01, wt_dec=5e-4),
        steps_per_epoch=1171,
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in model.rsbr.parameters()}
    assert all(
        id(parameter) not in optimizer_ids
        for name, parameter in model.named_parameters()
        if not name.startswith("rsbr.")
    )


def test_runner_has_hard_three_epoch_limit_and_no_epochs_argument():
    source = (ROOT / "tools" / "run_rsbr_v0_stage1_3ep.py").read_text(encoding="utf-8")
    assert EPOCHS == 3
    assert 'add_argument("--epochs"' not in source
    assert '"auto_continued": False' in source
    assert '"test_accessed": False' in source
    assert '"luad_accessed": False' in source


def test_paired_evaluator_is_update_free_and_contract_locked():
    source = (ROOT / "tool" / "infer_rsbr_v0_paired.py").read_text(encoding="utf-8")
    for forbidden in (
        "torch.optim", "PolyOptimizer", ".step(", ".backward(",
        "Stage1_TrainDataset", ".train()",
    ):
        assert forbidden not in source
    assert "0.6 * normalized_28_1" in source
    assert "0.2 * normalized_28_2" in source
    assert "0.2 * normalized_deep" in source
    assert "[0.8, 0.9, 0.8, 0.6]" in source
    assert "(((), ()), ((3,), (2,)), ((2,), (1,)))" in source


def test_scope_rejects_test_or_luad_paths(tmp_path):
    from tool.infer_rsbr_v0_paired import _validate_scope

    with pytest.raises(ValueError):
        _validate_scope(tmp_path / "test", ("base", "full"))
    with pytest.raises(ValueError):
        _validate_scope(tmp_path / "luad" / "val", ("base", "full"))


def test_server_preflight_is_disposable_and_checks_frozen_state():
    source = (ROOT / "tools" / "preflight_rsbr_v0_stage1.py").read_text(
        encoding="utf-8"
    )
    assert "RSBR_STAGE1_PREFLIGHT_PASS" in source
    assert "frozen_parameters_unchanged" in source
    assert "frozen_buffers_unchanged" in source
    assert "torch.save" not in source
    assert "paired_rsbr_validation" not in source
