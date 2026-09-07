from tools.fqrf_diagnostics import apply_epoch2_screen, apply_final_gate
from tools.fqrf_report import render_report
import torch

from train_fqrf_phase0 import CONFIG, PHASE0_STEPS, save_visualizations


def _healthy():
    return {
        "stagewise_mask_area": [{"stage": 3, "local_median": .4, "local_fraction_gt_090": .1, "global_median": .3}],
        "stagewise_query_redundancy": [{"stage": 3, "pair_iou_mean": .3, "pair_iou_median": .5, "pair_iou_fraction_gt_090": .1, "pair_iou_fraction_gt_075": .2}],
        "stagewise_embedding_diversity": [{"stage": i, "cosine_mean": .3, "cosine_median": .3, "cosine_p90": .5, "cosine_fraction_gt_095": .1} for i in (1, 2, 3)],
        "dynamic_focus": [{"stage": i, "active_fraction": .9, "displacement_median": 1, "focus_cosine_p90": .5} for i in (2, 3)],
        "attention_focus": [{"stage": i, "centroid_pairwise_distance_mean": .2} for i in (1, 2, 3)],
        "masked_attention_health": [{"stage": i, "visible_ratio_mean": .5, "fallback_to_global_fraction": .1} for i in (2, 3)],
        "semantic_selectivity": {"rival_leakage": .1, "background_leakage": .02, "positive_negative_gap": 12},
        "pca_health": [{"stage": 3, "present_class_query_coverage": .96, "absent_class_dominance": .05, "all_finite": True, "top1_confidence": .1}],
        "pmec_health": {"candidate_coverage": .91, "differs_from_top1": .8, "groups_per_pair": 4},
        "chpf_health": {}, "all_finite": True,
    }


def test_phase0_runtime_contract():
    assert PHASE0_STEPS == 3513 and CONFIG["epochs_max"] == 3 and CONFIG["validation_access"] is False
    assert CONFIG["full25_schedule_denominator"] == 29275 and CONFIG["attention_mask"]["threshold"] == .15


def test_epoch2_redundancy_boundary_is_catastrophic():
    value = _healthy(); red = value["stagewise_query_redundancy"][0]
    red.update(pair_iou_median=.95, pair_iou_fraction_gt_090=.60)
    assert apply_epoch2_screen(value)["decision"] == "FQRF_FOCUS_NOGO"


def test_epoch2_requires_both_redundancy_conditions():
    value = _healthy(); value["stagewise_query_redundancy"][0].update(pair_iou_median=.95, pair_iou_fraction_gt_090=.59)
    assert apply_epoch2_screen(value)["decision"] == "CONTINUE_EPOCH3"


def test_epoch2_fallback_strictly_over_half():
    value = _healthy(); value["masked_attention_health"][0]["fallback_to_global_fraction"] = .50
    assert apply_epoch2_screen(value)["decision"] == "CONTINUE_EPOCH3"
    value["masked_attention_health"][0]["fallback_to_global_fraction"] = .51
    assert apply_epoch2_screen(value)["decision"] == "FQRF_FOCUS_NOGO"


def test_strong_go_gate():
    assert apply_final_gate(_healthy())["decision"] == "FQRF_PHASE0_STRONG_GO"


def test_regular_go_gate():
    value = _healthy(); value["semantic_selectivity"]["positive_negative_gap"] = 8
    assert apply_final_gate(value)["decision"] == "FQRF_PHASE0_GO"


def test_final_redundancy_threshold_is_strict():
    value = _healthy(); value["stagewise_query_redundancy"][0]["pair_iou_median"] = .75
    assert apply_final_gate(value)["decision"] == "FQRF_FOCUS_NOGO"
    value = _healthy(); value["stagewise_query_redundancy"][0]["pair_iou_fraction_gt_090"] = .30
    assert apply_final_gate(value)["decision"] == "FQRF_FOCUS_NOGO"


def test_inclusive_pca_and_pmec_boundaries():
    value = _healthy(); pca = value["pca_health"][0]; pmec = value["pmec_health"]
    pca.update(present_class_query_coverage=.93, absent_class_dominance=.10); pmec.update(candidate_coverage=.85, differs_from_top1=.70)
    assert apply_final_gate(value)["decision"] in ("FQRF_PHASE0_GO", "FQRF_PHASE0_STRONG_GO")


def test_report_exact_last_line(tmp_path):
    summary = _healthy()
    path = render_report(tmp_path, {"decision": "FQRF_PHASE0_GO", "final_summary": summary, "gate": {"checks": {}}, "all_finite": True})
    assert path.read_text(encoding="utf-8").splitlines()[-1] == "DECISION = FQRF_PHASE0_GO"


def test_visualization_path(tmp_path):
    stages = []
    for _ in range(3):
        stages.append({
            "confidence": {"joint": torch.rand(1, 196, 4), "p_class": torch.rand(1, 196, 4)},
            "mask_logits": torch.randn(1, 196, 56, 56),
            "attention": {"cross_attention": torch.softmax(torch.randn(1, 196, 28 * 28), dim=-1)},
            "memory_hw": (28, 28),
        })
    output = {
        "normalized_cam": torch.rand(1, 4, 28, 28), "stages": stages,
        "pmec_region": torch.rand(1, 4, 56, 56),
    }
    save_visualizations(tmp_path, "unit", ["sample"], torch.zeros(1, 3, 224, 224), torch.tensor([[1., 0., 0., 0.]]), output)
    assert (tmp_path / "visualizations" / "unit" / "sample.png").stat().st_size > 0
