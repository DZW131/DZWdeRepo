from pathlib import Path
import hashlib

from tools.rsbr_parity_r1_contract import (
    MIOU_ALLOWANCE_PP,
    MODEL_IDENTITY_NOGO,
    NUMERICAL_REVIEW,
    PARITY_PASS,
    PIXEL_ALLOWANCE,
    decide_parity_r1,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    "network/rsbr_v0.py": "c4cf64a5dd564aa3dbbf7fe24b8aab672f0c1b964273848cdb34d76601f1c1d0",
    "network/resnet38_cls_rsbr.py": "c6a5281e3c047a2b299c4bf111732b23db7fd60ea11f9257a7b6e74414b00127",
}


def identity_kwargs():
    return {
        "max_cam_difference": 0.0,
        "delta_core_exact_zero": True,
        "delta_transition_exact_zero": True,
        "same_process_prediction_differences": 0,
    }


def test_miou_boundary_and_equality_are_pass():
    assert decide_parity_r1(
        **identity_kwargs(),
        production_miou_difference_pp=MIOU_ALLOWANCE_PP,
        production_prediction_differences=PIXEL_ALLOWANCE,
    ) == PARITY_PASS


def test_miou_threshold_plus_epsilon_fails():
    assert decide_parity_r1(
        **identity_kwargs(),
        production_miou_difference_pp=MIOU_ALLOWANCE_PP + 1e-12,
        production_prediction_differences=0,
    ) == NUMERICAL_REVIEW


def test_pixel_threshold_plus_epsilon_fails():
    assert decide_parity_r1(
        **identity_kwargs(),
        production_miou_difference_pp=0.0,
        production_prediction_differences=PIXEL_ALLOWANCE + 1,
    ) == NUMERICAL_REVIEW


def test_same_process_identity_is_a_hard_gate():
    values = identity_kwargs()
    values["max_cam_difference"] = 1e-12
    assert decide_parity_r1(
        **values,
        production_miou_difference_pp=0.0,
        production_prediction_differences=0,
    ) == MODEL_IDENTITY_NOGO


def test_model_sources_are_unchanged():
    for relative, expected in FROZEN_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_parity_audit_exposes_no_training_path():
    source = (ROOT / "tools/audit_rsbr_v0_parity_r1.py").read_text(encoding="utf-8")
    forbidden = ("optimizer", ".backward(", "Stage1_TrainDataset", ".train()")
    assert all(token not in source for token in forbidden)


def test_production_flags_remain_the_frozen_nondeterministic_protocol():
    source = (ROOT / "tools/audit_rsbr_v0_parity_r1.py").read_text(encoding="utf-8")
    required = (
        "torch.use_deterministic_algorithms(False)",
        "torch.backends.cudnn.benchmark = True",
        "torch.backends.cudnn.deterministic = False",
        'torch.backends.cuda.matmul.fp32_precision = "tf32"',
        'torch.backends.cudnn.conv.fp32_precision = "tf32"',
    )
    assert all(token in source for token in required)
