from tools.hqrf_diagnostics import apply_epoch2_screen, apply_final_gate
from tools.hqrf_report import render_report
from train_hqrf_phase0 import CONFIG, FULL25_STEPS, PHASE0_STEPS


def _healthy():
    return {
        "mask_area":{"local_fraction_gt_090":.1,"local_median":.4},
        "query_diversity":{"pair_iou_mean":.3,"pair_iou_fraction_gt_090":.1},
        "semantic_selectivity":{"rival_seed_leakage":.3,"background_leakage":.4,"positive_negative_logit_gap":1.2},
        "pca_health":{"present_pair_query_coverage":.99,"absent_confidence_dominance_image_fraction":.01,"all_finite":True},
        "pmec_health":{"candidate_pair_fraction":.9,"differs_from_top1_fraction":.6},
    }


def test_phase0_not_full25_training():
    assert PHASE0_STEPS==3513 and CONFIG["epochs_max"]==3 and CONFIG["validation_access"] is False


def test_schedule_denominator_is_full25():
    assert CONFIG["full25_schedule_denominator"]==FULL25_STEPS==29275


def test_epoch2_catastrophic_screen_strictly_greater_than_80_percent():
    summary={"catastrophic":{"snapshot":"epoch2","a":.80,"b":.81}}
    result=apply_epoch2_screen(summary); assert result["decision"]=="HQRF_QUERY_REGION_NOGO" and result["failed_criteria"]==["b"]


def test_strong_go_gate():
    assert apply_final_gate(_healthy())["decision"]=="HQRF_PHASE0_STRONG_GO"


def test_go_gate():
    value=_healthy(); value["semantic_selectivity"]["positive_negative_logit_gap"]=.7
    assert apply_final_gate(value)["decision"]=="HQRF_PHASE0_GO"


def test_failed_gate_is_nogo():
    value=_healthy(); value["mask_area"]["local_median"]=.8
    assert apply_final_gate(value)["decision"]=="HQRF_QUERY_REGION_NOGO"


def test_report_exact_last_line(tmp_path):
    summary=_healthy(); summary.update({"mask_logits":{},"chpf":{}}); summary["mask_area"].update({"global_median":.4}); summary["query_diversity"].update({"pair_iou_median":.2,"embedding_cosine_p90":.5}); summary["semantic_selectivity"].update({"mean_positive_logit":1.,"mean_negative_logit":0.}); summary["pca_health"].update({"top1_confidence":.1}); summary["pmec_health"].update({"region_groups_per_pair":2.})
    path=render_report(tmp_path,{"decision":"HQRF_PHASE0_GO","final_summary":summary,"gate":{"checks":{}},"all_finite":True})
    assert path.read_text(encoding="utf-8").splitlines()[-1]=="DECISION = HQRF_PHASE0_GO"
