from pathlib import Path

from tools.cqrf_diagnostics import apply_epoch2_screen, apply_final_gate
from tools.cqrf_report import render_report
from train_cqrf_phase0 import CONFIG, PHASE0_STEPS, SNAPSHOTS


def _summary(snapshot="epoch5"):
    return {"stagewise_mask_area":[{"stage":3,"local_median":.4,"local_fraction_gt_090":.1,"local_fraction_lt_005":.1}],"stagewise_query_redundancy":[{"stage":3,"pair_iou_median":.5,"pair_iou_fraction_gt_090":.1,"pair_iou_fraction_gt_075":.2}],"responsibility_utilization":[{"stage":3,"dominant_share_median":.4,"effective_queries_median":3}],"responsibility_integrity":[{"stage":3,"max_abs_sum_error":1e-7}],"semantic_selectivity":{"rival_leakage":.1,"background_leakage":.02,"positive_negative_gap":12},"pca_health":[{"stage":3,"present_class_query_coverage":.96,"absent_class_dominance":.05,"all_finite":True}],"pmec_health":{"candidate_coverage":.91,"differs_from_top1":.8},"all_finite":True,"snapshot":snapshot}


def _history():
    values=[]
    for epoch,median,high in ((3,.48,.10),(4,.49,.11),(5,.50,.12)):
        s=_summary(f"epoch{epoch}"); s["stagewise_query_redundancy"][0].update(pair_iou_median=median,pair_iou_fraction_gt_090=high); values.append(s)
    return values


def test_runtime_contract():
    assert PHASE0_STEPS==5855 and CONFIG["epochs_max"]==5 and CONFIG["full25_schedule_denominator"]==29275
    assert SNAPSHOTS[2342]=="epoch2" and SNAPSHOTS[5855]=="epoch5" and CONFIG["ccra"]["softmax_dimension"]=="query"


def test_epoch2_competition_and_monopoly_boundaries():
    s=_summary("epoch2"); s["stagewise_query_redundancy"][0].update(pair_iou_median=.93,pair_iou_fraction_gt_090=.55)
    assert apply_epoch2_screen(s)["decision"]=="CQRF_CCRA_NOGO"
    s=_summary("epoch2"); s["responsibility_utilization"][0].update(dominant_share_median=.91,effective_queries_median=1.49)
    assert apply_epoch2_screen(s)["decision"]=="CQRF_CCRA_NOGO"


def test_epoch2_requires_joint_conditions():
    s=_summary("epoch2"); s["stagewise_query_redundancy"][0].update(pair_iou_median=.93,pair_iou_fraction_gt_090=.54)
    assert apply_epoch2_screen(s)["decision"]=="CONTINUE_TO_EPOCH5"


def test_strong_and_regular_go():
    assert apply_final_gate(_history())["decision"]=="CQRF_PHASE0_STRONG_GO"
    values=_history(); values[-1]["semantic_selectivity"]["positive_negative_gap"]=8
    assert apply_final_gate(values)["decision"]=="CQRF_PHASE0_GO"


def test_rebound_and_inclusive_boundary_fail():
    values=_history(); values[-1]["stagewise_query_redundancy"][0]["pair_iou_median"]=.56
    assert apply_final_gate(values)["decision"]=="CQRF_CCRA_NOGO"
    values=_history(); values[-1]["responsibility_utilization"][0]["dominant_share_median"]=.75
    assert apply_final_gate(values)["decision"]=="CQRF_CCRA_NOGO"


def test_report_has_43_sections_and_exact_last_line(tmp_path):
    result={"decision":"CQRF_PHASE0_GO","final_summary":_summary(),"gate":{"checks":{}},"all_finite":True}
    path=render_report(tmp_path,result); text=path.read_text(encoding="utf-8")
    assert text.count("\n## ")==43 and text.splitlines()[-1]=="DECISION = CQRF_PHASE0_GO"


def test_history_snapshot_is_inferred_from_nested_rows():
    values=_history()
    for value in values:
        value["stagewise_query_redundancy"][0]["snapshot"]=value["snapshot"]
        value.pop("snapshot")
    assert apply_final_gate(values)["decision"]=="CQRF_PHASE0_STRONG_GO"
