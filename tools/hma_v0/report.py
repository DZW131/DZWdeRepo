"""Markdown report renderer for SSHR HMA-v0."""

from __future__ import annotations

from pathlib import Path

from tools.hma_v0 import LOSS_WEIGHTS, STAGES


def _pct(value):
    return f"{100.0 * value:.4f}"


def _num(value):
    return f"{value:.6g}"


def _stage_impact(validation, stage):
    full = validation["final_variants"]["official_full"]["mean_iou"]
    off = validation["final_variants"][f"hfrm_{stage}_off"]["mean_iou"]
    return full - off


def render_report(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = summary["provenance"]
    validation = summary["validation"]
    gradient = summary["gradient"]
    gamma = summary["gamma"]
    kernels = summary["kernels"]
    mechanism = summary["mechanism_map"]
    final = validation["final_variants"]
    standalone = validation["standalone_cam"]
    pipeline = validation["pipeline_decomposition"]
    response = validation["gsr_response"]
    spatial = validation["ch_spatial_effect"]
    comp = validation["complementarity"]
    errors = validation["error_taxonomy"]
    present = validation["present_confusion"]

    raw = final["all_hfrm_off"]["mean_iou"]
    full = final["official_full"]["mean_iou"]
    gsr = final["gsr_only"]["mean_iou"]
    ch = final["ch_only"]["mean_iou"]
    absent = response["absent_primary"]["all_stages"]
    present_raw = sum(item["raw_present_confusion"] for item in present.values())
    present_net = sum(item["net"] for item in present.values())
    present_rate = present_net / max(present_raw, 1)
    kernel_behavior = ", ".join(f"{s}: {kernels[s]['behavior']}" for s in STAGES)
    boundary = spatial["raw_to_ch"]["B0_le_2"]
    interior = spatial["raw_to_ch"]["B2_ge_8"]
    gate_gain = pipeline["full_official_gate"]["mean_iou"] - pipeline["full_no_gate"]["mean_iou"]
    norm = response["normalization"]["all_stages"]
    impact_1 = _stage_impact(validation, "28_1")
    impact_2 = _stage_impact(validation, "28_2")
    dominant_stage = "28_1" if impact_1 > impact_2 else "28_2" if impact_2 > impact_1 else "tie"
    deep_grad = gradient["feat_deep_gradient_norm"]["deep"]["mean"]
    shallow_grad = sum(gradient["feat_deep_gradient_norm"][s]["mean"] for s in STAGES)
    remaining_absent = errors["official_full"]["absent_class"]["candidate_wrong"]
    remaining_present = errors["official_full"]["present_confusion"]["candidate_wrong"]
    error_dominant = "present-class confusion" if remaining_present > remaining_absent else "absent-class error"

    lines = [
        "# SSHR-HMA-v0 — HFRM Mechanism Autopsy",
        "",
        "> 本报告是 frozen-checkpoint 机制审计，不是模型改造或调参实验。全程未构造 optimizer、未调用 optimizer.step、未评估 BCSS test/LUAD，也未训练新模型。",
        "",
        "## 1. Frozen provenance and safety",
        "",
        f"- A0 commit: `{provenance['a0_commit']}`",
        f"- Audit commit: `{provenance['audit_commit']}`",
        f"- Checkpoint: `{provenance['checkpoint']}`",
        f"- Checkpoint SHA256: `{provenance['checkpoint_sha256']}`",
        f"- BCSS validation: {provenance['validation_images']} images / {provenance['validation_slides']} slides",
        f"- Fixed gradient audit: {gradient['batches']} batches × {gradient['batch_size']}, seed={gradient['seed']}",
        f"- Parameter SHA before/after gradient audit: `{gradient['parameter_hash_before']}` / `{gradient['parameter_hash_after']}`",
        f"- Buffer SHA before/after gradient audit: `{gradient['buffer_hash_before']}` / `{gradient['buffer_hash_after']}`",
        f"- Runtime: {provenance['runtime_seconds'] / 60:.2f} min; peak CUDA memory: {provenance['peak_cuda_memory_gib']:.3f} GiB",
        "",
        "### Instrumentation hard parity",
        "",
        f"Decision: **{summary['parity']['decision']}**. "
        f"Compared {summary['parity']['tensors_compared']} same-process tensors over "
        f"{summary['parity']['images']} images × {summary['parity']['tta_views_per_image']} TTA views; "
        f"max absolute difference={summary['parity']['maximum_absolute_difference']:.1f}, "
        f"final differing pixels={summary['parity']['final_prediction_differing_pixels']}.",
        "",
        "## 2. Executive summary",
        "",
        "| Question | Measurement | Result | Evidence Level |",
        "|---|---|---|---|",
        f"| GSR truly vetoes? | absent response Δ | median Δlogit={_num(absent['median_delta_logit'])}; suppressed={100*absent['fraction_suppressed']:.2f}% | Direct frozen measurement |",
        f"| GSR handles present confusion? | present-confusion net recovery | {present_net:,}/{present_raw:,} ({100*present_rate:.3f}%) | Direct frozen measurement |",
        f"| CH remains low-pass? | kernel FFT / uniform cosine | {kernel_behavior} | Direct parameter measurement |",
        f"| CH helps interior? | raw→CH B2 | net={interior['net']:,}; Δacc={100*interior['accuracy_delta']:.4f} pp | Direct causal measurement |",
        f"| CH hurts boundary? | raw→CH B0 | net={boundary['net']:,}; Δacc={100*boundary['accuracy_delta']:.4f} pp | Direct causal measurement |",
        f"| GSR/CH complementary? | unique recover / Jaccard | G unique={comp['gsr_unique_recover']:,}; CH unique={comp['ch_unique_recover']:,}; J={comp['recovery_set_jaccard']:.4f} | Direct paired prediction |",
        f"| 28_1 or 28_2 dominant? | paired branch-off ΔmIoU | 28_1={100*impact_1:+.4f} pp; 28_2={100*impact_2:+.4f} pp; {dominant_stage} larger | Direct causal measurement |",
        f"| deep supervision dominant? | feat_deep gradient norm | deep={_num(deep_grad)}; shallow sum={_num(shallow_grad)} | Fixed-batch gradient audit |",
        f"| class gate contributes strongly? | gated vs ungated | ΔmIoU={100*gate_gain:+.4f} pp | Direct pipeline decomposition |",
        f"| min-max amplifies response? | raw vs normalized delta | median ratio={norm['amplification_ratio_median']:.3f}; >2×={100*norm['fraction_amplified_over_2x']:.2f}% | Direct response measurement |",
        "",
        "## 3. Learned scalar autopsy",
        "",
        "| Stage | gamma_veto | gamma_context | sign_veto | sign_context | |veto/context| |",
        "|---|---:|---:|---|---|---:|",
    ]
    for stage in STAGES:
        item = gamma[stage]
        lines.append(
            f"| {stage} | {_num(item['gamma_veto'])} | {_num(item['gamma_context'])} | "
            f"{item['sign_veto']} | {item['sign_context']} | {_num(item['absolute_veto_context_ratio'])} |"
        )
    lines += [
        "",
        "按公开公式，`gamma_veto > 0` 时 GSR 项是对 gated feature 的加性放大/调制；只有 `gamma_veto < 0` 才是直接 feature attenuation。语义上的 absent-class 抑制需由实测响应另行判断，不能仅凭模块命名断言。",
        "",
        "## 4. Context-kernel autopsy",
        "",
        "| Stage | Uniform Cosine | Neg. Fraction | DC Gain | HF/LF | Anisotropy | Label |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for stage in STAGES:
        item = kernels[stage]
        lines.append(
            f"| {stage} | {_num(item['uniform_cosine']['median'])} | "
            f"{_num(item['negative_fraction']['median'])} | {_num(item['dc_gain']['median'])} | "
            f"{_num(item['hf_lf_ratio']['median'])} | {_num(item['anisotropy']['median'])} | {item['behavior']} |"
        )
    lines += [
        "",
        "K=15 对应约 60 input pixels（F56）和 120 input pixels（F28）；这是尺度结构事实，不构成 K 值优劣的性能证据。",
        "",
        "## 5. Same-forward causal CAM audit",
        "",
        "| Branch | Raw mIoU | GSR-only | CH-only | Full | GSR Gain | CH Gain | Full Gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        values = standalone[stage]
        stage_raw = values["raw"]["mean_iou"]
        lines.append(
            f"| CAM{stage} | {_pct(stage_raw)} | {_pct(values['gsr']['mean_iou'])} | "
            f"{_pct(values['ch']['mean_iou'])} | {_pct(values['full']['mean_iou'])} | "
            f"{100*(values['gsr']['mean_iou']-stage_raw):+.4f} | "
            f"{100*(values['ch']['mean_iou']-stage_raw):+.4f} | "
            f"{100*(values['full']['mean_iou']-stage_raw):+.4f} |"
        )
    deep = standalone["deep"]["raw"]["mean_iou"]
    lines += [
        f"| CAMdeep | {_pct(deep)} | — | — | {_pct(deep)} | — | — | 0.0000 |",
        "",
        "CAM56 的结果只描述其 standalone head；released final fusion 不使用 CAM56，而且 frozen ablation 不能识别其历史训练期因果贡献。",
        "",
        "### Final official-fusion variants",
        "",
        "| Variant | mIoU | Δ vs Full | mDice | C0 | C1 | C2 | C3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    variant_labels = {
        "official_full": "Official Full", "all_hfrm_off": "All HFRM off",
        "gsr_only": "GSR-only", "ch_only": "CH-only",
        "hfrm_28_1_off": "28_1 HFRM off", "hfrm_28_2_off": "28_2 HFRM off",
        "gsr_28_1_off": "28_1 GSR off", "ch_28_1_off": "28_1 CH off",
        "gsr_28_2_off": "28_2 GSR off", "ch_28_2_off": "28_2 CH off",
    }
    for key, label in variant_labels.items():
        item = final[key]
        delta = "—" if key == "official_full" else f"{100*(item['mean_iou']-full):+.4f}"
        classes = " | ".join(_pct(item["class_iou"][str(c)]) for c in range(4))
        lines.append(f"| {label} | {_pct(item['mean_iou'])} | {delta} | {_pct(item['mean_dice'])} | {classes} |")
    lines += [
        "",
        f"Full vs Raw frozen-checkpoint effect: **{100*(full-raw):+.4f} mIoU points**. "
        f"GSR-only gain={100*(gsr-raw):+.4f}; CH-only gain={100*(ch-raw):+.4f}.",
        "",
        "## 6. GSR, CH, and inference pipeline",
        "",
        f"- Absent-class response: median GSR Δlogit={_num(absent['median_delta_logit'])}, mean={_num(absent['mean_delta_logit'])}, suppression fraction={100*absent['fraction_suppressed']:.2f}%.",
        f"- Present-confusion net recovery: {present_net:,} pixels from {present_raw:,} raw present-confusion errors ({100*present_rate:.3f}%).",
        f"- CH raw→CH near-boundary net={boundary['net']:,}, interior net={interior['net']:,}; paired GSR→Full results are retained in `paired_causal/ch_spatial_effect.json`.",
        f"- Complementarity: recover Jaccard={comp['recovery_set_jaccard']:.4f}; GSR-only unique={comp['gsr_unique_recover']:,}; CH-only unique={comp['ch_unique_recover']:,}; overlap={comp['both_recover']:,}.",
        f"- Official hard gate: Full no-gate={_pct(pipeline['full_no_gate']['mean_iou'])}, Full gated={_pct(pipeline['full_official_gate']['mean_iou'])}, Δ={100*gate_gain:+.4f} points.",
        f"- Min-max response: median |raw-scale delta|={_num(norm['range_scaled_raw_delta_abs_median'])}; median normalized/raw amplification ratio={norm['amplification_ratio_median']:.3f}; fraction >2×={100*norm['fraction_amplified_over_2x']:.2f}%.",
        "",
        "## 7. Fixed training-gradient audit",
        "",
        "| Loss branch | Weight | Shared Early | Mid | Late | feat_deep | HFRM target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    target_group = {"56": "hfrm_56", "28_1": "hfrm_28_1", "28_2": "hfrm_28_2"}
    for branch in ("56", "28_1", "28_2", "deep"):
        groups = gradient["group_summary"][branch]
        target = "—" if branch == "deep" else _num(groups[target_group[branch]]["gradient_norm"]["mean"])
        lines.append(
            f"| L{branch} | {LOSS_WEIGHTS[branch]:.2f} | "
            f"{_num(groups['shared_early']['gradient_norm']['mean'])} | "
            f"{_num(groups['mid']['gradient_norm']['mean'])} | "
            f"{_num(groups['late']['gradient_norm']['mean'])} | "
            f"{_num(gradient['feat_deep_gradient_norm'][branch]['mean'])} | {target} |"
        )
    lines += [
        "",
        f"Direct deep-loss feat_deep gradient norm={_num(deep_grad)}; the three shallow losses' norms sum to {_num(shallow_grad)} (ratio={shallow_grad/(deep_grad+1e-20):.4f}). "
        "Gradient-cosine matrices are in the JSON audit and `figures/gradient_cosine_matrix.png`; these are fixed-batch observational gradients, not optimizer updates.",
        "",
        "## 8. Error taxonomy",
        "",
        "| Candidate | Error type | Raw wrong | Candidate wrong | Recovered | Harmed | Net |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in ("gsr_only", "ch_only", "official_full"):
        for category in ("absent_class", "present_confusion", "boundary", "interior"):
            item = errors[candidate][category]
            lines.append(
                f"| {candidate} | {category} | {item['raw_wrong']:,} | {item['candidate_wrong']:,} | "
                f"{item['recovered']:,} | {item['harmed']:,} | {item['net']:,} |"
            )
    lines += [
        "",
        f"在 Full frozen prediction 中，remaining absent-class={remaining_absent:,}，present-class confusion={remaining_present:,}；数量更大的类别是 **{error_dominant}**。",
        "",
        "## 9. Evidence-ranked weakness map",
        "",
        f"Completion: **{mechanism['completion']}**",
        "",
        "Evidence labels: " + ", ".join(f"`{label}`" for label in mechanism["labels"]),
        "",
        "### Tier A — directly supported",
        "",
    ]
    lines += [f"- {item}" for item in mechanism["tier_a_directly_supported"]]
    lines += ["", "### Tier B — structurally plausible", ""]
    lines += [f"- {item}" for item in mechanism["tier_b_structurally_plausible"]]
    lines += ["", "### Tier C — speculative", ""]
    lines += [f"- {item}" for item in mechanism["tier_c_speculative"]]
    lines += [
        "",
        "## 10. Answers to the 20 frozen questions",
        "",
        f"1. Gamma values/signs are listed in Section 3; exact values are preserved in `parameter_autopsy/gamma_autopsy.json`.",
        f"2. By sign, trained GSR is mathematically additive amplification/modulation; empirical absent response is a separate measurement.",
        f"3. Absent classes: median Δlogit={_num(absent['median_delta_logit'])}, suppressed fraction={100*absent['fraction_suppressed']:.2f}%; this {'meets' if mechanism['measurements']['veto_supported'] else 'does not meet'} the frozen veto criterion.",
        f"4. Present confusion net recovery={present_net:,} pixels ({100*present_rate:.3f}% of the raw present-confusion count).",
        f"5. Trained CH labels: {kernel_behavior}; the conclusion uses direct spatial weights and zero-padded FFT, not module naming.",
        f"6. CH raw→CH boundary Δacc={100*boundary['accuracy_delta']:+.4f} pp; interior Δacc={100*interior['accuracy_delta']:+.4f} pp.",
        f"7. Boundary harm is {'supported' if boundary['net'] < 0 else 'not supported'} by the preregistered net-transition sign (net={boundary['net']:,}).",
        f"8. GSR/CH recover-set Jaccard={comp['recovery_set_jaccard']:.4f}; the frozen label is " + ("complementary." if "GSR_CH_COMPLEMENTARY" in mechanism["labels"] else "redundant." if "GSR_CH_REDUNDANT" in mechanism["labels"] else "conflicting/mixed by the declared rule."),
        f"9. Direct official-fusion removal costs: 28_1={100*impact_1:+.4f} pp, 28_2={100*impact_2:+.4f} pp; {dominant_stage} is larger.",
        f"10. CAM56 standalone Raw/GSR/CH/Full is reported in Section 5. It is excluded from released fusion, and frozen inference cannot establish its training-time causal contribution.",
        f"11. Same-checkpoint Full−Raw={100*(full-raw):+.4f} mIoU points.",
        f"12. Official hard class-gate contribution on Full={100*gate_gain:+.4f} mIoU points.",
        f"13. Min-max median amplification ratio={norm['amplification_ratio_median']:.3f}; {100*norm['fraction_amplified_over_2x']:.2f}% of stage/class/image cells exceed 2×.",
        f"14. Deep weighted loss feat_deep norm={_num(deep_grad)} vs shallow sum={_num(shallow_grad)}; dominance is decided from these measured norms, not nominal 0.50 alone.",
        f"15. Shallow-to-deep gradient ratio={shallow_grad/(deep_grad+1e-20):.4f}; per-branch values and cosines are archived.",
        f"16. Remaining error is dominated by {error_dominant}: absent={remaining_absent:,}, present-confusion={remaining_present:,}.",
        "17. Tier-A weaknesses are exactly those listed in Section 9 and have direct parameter/paired-prediction/gradient evidence.",
        "18. Tier-B items are structural facts or plausible bottlenecks without frozen causal performance proof.",
        "19. Tier-C items are deliberately retained as untested hypotheses and are not promoted to innovation designs.",
        f"20. Most important unresolved scientific question: **{mechanism['single_most_important_unresolved_scientific_question']}**",
        "",
        "## 11. Artifact map",
        "",
        "- `provenance/manifest.json` and `provenance/source_contract.json`",
        "- `parameter_autopsy/gamma_autopsy.json`",
        "- `kernels/kernel_channel_metrics.csv` and `kernels/kernel_summary.json`",
        "- `gates/gate_vectors.npz`, gate statistics, semantic separability, and GSR response rows",
        "- `paired_causal/final_variants.json`, CH spatial effects, complementarity, and present-confusion effects",
        "- `standalone_cam/standalone_cam.json`",
        "- `error_taxonomy/error_taxonomy.json`",
        "- `inference_decomposition/pipeline.json`",
        "- `gradient_audit/gradient_rows.csv`, component rows, and summary",
        "- `figures/*.png`",
        "",
        "---",
        "",
        "**HFRM_MECHANISM_MAP_COMPLETE**",
        "",
        "STOP: no new HFRM, gamma/K change, spatial gate, loss, training, test, or LUAD evaluation was executed.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
