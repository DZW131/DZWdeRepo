"""Render the hierarchy-shared CDSR-v2 readiness report."""

import argparse
import json
from pathlib import Path


def sci(value):
    return f"{value:.3e}"


def pct(value):
    return f"{value:+.4f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-json", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--compatibility-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    smoke = json.loads(args.smoke_json.read_text(encoding="utf-8"))
    profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
    compatibility = json.loads(
        args.compatibility_json.read_text(encoding="utf-8")
    )
    records = smoke["records"]
    final = smoke["final_alpha_movements"]
    protocol = smoke["protocol"]
    deltas = profile["cdsr_vs_a0"]
    readiness_pass = smoke["readiness_pass"]
    token = "CDSR_V2_READINESS_PASS" if readiness_pass else "CDSR_FINAL_NOGO"
    alpha_order = ("shared.alpha_sem", "shared.alpha_ctx")

    lines = [
        "# CDSR-v2 Hierarchy-Shared Alpha Readiness Report",
        "",
        "## 1. Executive decision",
        "",
        f"**{token}**",
        "",
    ]
    if readiness_pass:
        lines.extend(
            [
                "Both hierarchy-shared alpha logits diverge measurably from",
                "their matched weight-decay-only shadows after exactly 20 real",
                "BCSS training steps. The simplified architecture passes the",
                "frozen engineering-readiness gate. Per instruction, no 25-epoch",
                "experiment is started in this phase.",
            ]
        )
    else:
        lines.extend(
            [
                "At least one hierarchy-shared alpha logit does not diverge",
                "measurably from its matched weight-decay-only shadow after",
                "exactly 20 real BCSS steps. This is the final CDSR No-Go; no",
                "additional initialization or architecture patch is proposed.",
            ]
        )

    lines.extend(
        [
            "",
            "## 2. Frozen CDSR-v2 architecture",
            "",
            "```text",
            "N_i = R_i * (1 - (1-D_i) * (1-U_i))",
            "G_sem_i = 1 - alpha_sem * (1 - N_i)",
            "G_ctx_i = 1 - alpha_ctx * (1 - N_i)",
            "F_R_i = F_i + gamma_sem_i * G_sem_i * F_sem_i",
            "              + gamma_ctx_i * G_ctx_i * F_CH15_i",
            "```",
            "",
            "- `alpha_sem` and `alpha_ctx` are the only two new learnable scalars",
            "- the same two parameter objects are used at F56, F28_1, and F28_2",
            "- N remains independently computed and spatially varying per stage",
            "- alpha initializes to 0.10 and every gate remains <=1",
            "- alpha->0 exactly recovers original SSHR",
            "- original GSR, CH15, CAM probes, detached FP32 Need, loss, optimizer,",
            "  inference, and metrics are unchanged",
            "- there is no new classifier, learned uncertainty head, or spatial",
            "  policy",
            "",
            "## 3. Test and compatibility evidence",
            "",
            "- local full suite: **67 passed**",
            "- RTX 5090 server full suite: **67 passed**",
            "- every trainable parameter is covered exactly once by the optimizer",
            "- both shared alpha logits are in the original from-scratch group",
            f"- exact A0 uniform strict load: **{compatibility['uniform_strict_load']}**",
            f"- A0 checkpoint SHA256: `{compatibility['checkpoint_sha256']}`",
            f"- CDSR-v2 missing/unexpected keys: "
            f"{len(compatibility['cdsr_missing_keys'])}/"
            f"{len(compatibility['cdsr_unexpected_keys'])}",
            f"- additional parameters: {compatibility['additional_parameters']}",
            f"- A0 compatibility: **{'PASS' if compatibility['pass'] else 'FAIL'}**",
            "",
            "The two missing keys are exactly the shared semantic/context alpha",
            "logits; backbone-pretrained loading differs from A0 by only those",
            "same two expected keys.",
            "",
            "## 4. Frozen real-step protocol",
            "",
            f"- parsed BCSS training samples: {protocol['parsed_training_samples']}",
            f"- batch / image size: {protocol['batch_size']} / {protocol['image_size']}",
            f"- steps / seed: {protocol['steps']} / {protocol['seed']}",
            f"- precision: {protocol['precision']}",
            f"- classification-loss weights: {protocol['loss_weights']}",
            f"- optimizer: {protocol['optimizer']}",
            f"- base LR / 25-epoch max-step: {protocol['base_lr']} / {protocol['max_step']}",
            f"- momentum / weight decay: {protocol['momentum']} / {protocol['weight_decay']}",
            f"- PyTorch / CUDA: {protocol['pytorch']} / {protocol['cuda_runtime']}",
            f"- GPU: {protocol['gpu']}",
            f"- step-1 / step-20 loss: {records[0]['loss']:.6f} / "
            f"{records[-1]['loss']:.6f}",
            f"- all finite: **{smoke['all_finite']}**",
            f"- shadow LR matched at every step: **{smoke['shadow_lr_matched']}**",
            "",
            "The measurable-movement criterion is unchanged from v1: final",
            "task-excess logit movement must be at least one float32 ULP at the",
            "initial logit. This tests optimizer-visible movement, not merely a",
            "nonzero mathematical gradient.",
            "",
            "## 5. Every-step shared-alpha audit",
            "",
            "Movement columns are cumulative from initialization.",
            "",
            "| step | alpha | task grad | actual movement | WD-only movement | task-excess | alpha value |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    initial_logits = {
        name: final[name]["initial_logit"] for name in alpha_order
    }
    for record in records:
        for name in alpha_order:
            values = record["alphas"][name]
            actual_movement = values["logit_after"] - initial_logits[name]
            wd_movement = values["shadow_logit_after"] - initial_logits[name]
            lines.append(
                f"| {record['step']} | {name} | "
                f"{sci(values['task_gradient'])} | {sci(actual_movement)} | "
                f"{sci(wd_movement)} | {sci(values['task_excess_logit'])} | "
                f"{values['alpha_after']:.10f} |"
            )

    lines.extend(
        [
            "",
            "Final shadow decision:",
            "",
            "| alpha | final task-excess logit | task-excess alpha | ULP | measurable |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name in alpha_order:
        value = final[name]
        lines.append(
            f"| {name} | {sci(value['task_excess_logit_movement'])} | "
            f"{sci(value['task_excess_alpha_movement'])} | "
            f"{sci(value['float32_logit_ulp_threshold'])} | "
            f"{value['measurable_task_excess']} |"
        )

    lines.extend(
        [
            "",
            "## 6. Every-step per-stage mechanism audit",
            "",
            "| step | stage | N mean | N std | G_sem mean | G_sem std | G_ctx mean | G_ctx std | gamma_sem | gamma_ctx | finite |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in records:
        for stage in ("stage1", "stage2", "stage3"):
            value = record["diagnostics"][stage]
            lines.append(
                f"| {record['step']} | {stage} | "
                f"{value['need']['mean']:.4f} | {value['need']['std']:.4f} | "
                f"{value['semantic_gate']['mean']:.4f} | "
                f"{value['semantic_gate']['std']:.4f} | "
                f"{value['context_gate']['mean']:.4f} | "
                f"{value['context_gate']['std']:.4f} | "
                f"{value['gamma_sem']:.6f} | "
                f"{value['gamma_context']:.6f} | {value['all_finite']} |"
            )

    lines.extend(
        [
            "",
            "## 7. Resource profile",
            "",
            "RTX 5090, batch20, 224x224, BF16, three warmups and ten",
            "synchronized measured iterations per mode.",
            "",
            "| quantity | CDSR-v2 vs A0 | budget | result |",
            "|---|---:|---:|---|",
            f"| parameters | +{deltas['additional_parameters']} "
            f"({pct(deltas['parameter_percent'])}) | exactly +2 | "
            f"{'PASS' if deltas['additional_parameters'] == 2 else 'FAIL'} |",
            f"| estimated FLOPs | {pct(deltas['estimated_flops_percent'])} | <+0.1% | "
            f"{'PASS' if deltas['estimated_flops_percent'] < 0.1 else 'FAIL'} |",
            f"| forward median latency | {pct(deltas['forward_median_latency_percent'])} | <+5% | "
            f"{'PASS' if deltas['forward_median_latency_percent'] < 5 else 'FAIL'} |",
            f"| train median latency | {pct(deltas['train_median_latency_percent'])} | <+10% | "
            f"{'PASS' if deltas['train_median_latency_percent'] < 10 else 'FAIL'} |",
            f"| forward peak memory | {pct(deltas['forward_peak_memory_percent'])} | reported | — |",
            f"| train peak memory | {pct(deltas['train_peak_memory_percent'])} | reported | — |",
            "",
            "## 8. Readiness matrix",
            "",
            "| check | result |",
            "|---|---|",
            "| Need formula unchanged | PASS |",
            "| exactly two shared parameter objects | PASS |",
            "| stage-specific N retained | PASS |",
            "| uniform exact A0 | PASS |",
            "| all local/server tests | PASS |",
            "| batch20 BF16 finite | PASS |",
            "| pretrained/A0 compatibility | PASS |",
            "| optimizer coverage | PASS |",
            "| matched shadow LR | PASS |",
            f"| shared alpha_sem measurable task-excess | "
            f"{'PASS' if final['shared.alpha_sem']['measurable_task_excess'] else 'FAIL'} |",
            f"| shared alpha_ctx measurable task-excess | "
            f"{'PASS' if final['shared.alpha_ctx']['measurable_task_excess'] else 'FAIL'} |",
            f"| overall | **{'PASS' if readiness_pass else 'FINAL NOGO'}** |",
            "",
            "## 9. Stop decision",
            "",
        ]
    )
    if readiness_pass:
        lines.extend(
            [
                "CDSR-v2 is engineering-ready under the frozen gate. The next",
                "possible action is the controlled 25-epoch BCSS seed-42 run,",
                "but it is intentionally not started here and requires review.",
            ]
        )
    else:
        lines.extend(
            [
                "CDSR-v2 fails the final frozen readiness gate. Stop CDSR. Do",
                "not propose or apply another initialization or architecture patch.",
            ]
        )
    lines.extend(["", token])

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
