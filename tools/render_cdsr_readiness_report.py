"""Render the CDSR implementation-readiness report from frozen JSON evidence."""

import argparse
import json
from pathlib import Path


def percent(value):
    return f"{value:+.4f}%"


def scientific(value):
    return f"{value:.3e}"


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
    protocol = smoke["protocol"]
    records = smoke["records"]
    final = smoke["final_alpha_movements"]
    deltas = profile["cdsr_vs_a0"]
    last_diagnostics = records[-1]["diagnostics"]
    passed_alphas = sum(
        value["measurable_task_excess"] for value in final.values()
    )
    passed_excess_ratios = [
        100.0
        * abs(value["task_excess_logit_movement"])
        / max(abs(value["weight_decay_only_logit_movement"]), 1e-30)
        for value in final.values()
        if value["measurable_task_excess"]
    ]

    lines = [
        "# CDSR Implementation Readiness Report",
        "",
        "## 1. Executive decision",
        "",
        "**CDSR_IMPLEMENTATION_READINESS_FAIL**",
        "",
        "The frozen Full CDSR architecture is structurally correct, finite,",
        "A0-compatible, and within every resource budget. However, the mandatory",
        "20-real-step weight-decay-shadow audit passes for only 4/6 alpha scalars.",
        "Both F28_2 alpha logits remain bitwise identical to their matched",
        "weight-decay-only shadows. Therefore the frozen readiness gate fails and",
        "the 25-epoch BCSS experiment must not start.",
        "",
        "This report does not change the Need formula, remove F28_2, create",
        "class-specific behavior, tune D/U/R, alter alpha initialization, or",
        "modify the loss/optimizer.",
        "",
        "## 2. Frozen implementation",
        "",
        "```text",
        "N = R * (1 - (1-D) * (1-U))",
        "G_sem = 1 - alpha_sem * (1 - N)",
        "G_ctx = 1 - alpha_ctx * (1 - N)",
        "F_R = F + gamma_sem * G_sem * F_sem + gamma_ctx * G_ctx * F_CH15",
        "```",
        "",
        "- original GSR semantic content is unchanged",
        "- original CH15 context is unchanged",
        "- raw probes reuse `ic_56`, `ic1`, `ic2`, and `fc8` under no-grad",
        "- all D/U/R/N math is detached FP32 and exactly matches Phase 0",
        "- there is no new classifier, spatial predictor, uncertainty head, or loss",
        "- three stages each add semantic/context alpha logits: six scalars total",
        "- alpha initializes to 0.10; initial gates lie in [0.9, 1.0]",
        "- `uniform` remains the default exact SSHR path",
        "- CDSR rejects FA-MPR and archived HST combinations",
        "",
        "## 3. Test and integration evidence",
        "",
        "- local full suite: **66 passed**",
        "- RTX 5090 server full suite: **66 passed**",
        "- batch20 / 224 / BF16 real-data forward-backward: finite for 20/20 steps",
        "- optimizer coverage: every trainable parameter exactly once; all six",
        "  alpha logits are in the original from-scratch weight group",
        "- tested: uniform equivalence, frozen formula equality, JSD symmetry and",
        "  bounds, entropy/reliability/Need bounds, detached probe, shared CAM",
        "  heads, six-scalar count, alpha initialization, gate bounds, alpha=0",
        "  exact HFRM fallback, shapes, finite CAMs, CLI isolation, and optimizer",
        "  coverage",
        "",
        "## 4. A0 and pretrained compatibility",
        "",
        f"- A0 checkpoint: `{compatibility['checkpoint']}`",
        f"- SHA256: `{compatibility['checkpoint_sha256']}`",
        f"- exact uniform strict load: **{compatibility['uniform_strict_load']}**",
        f"- CDSR unexpected keys: {len(compatibility['cdsr_unexpected_keys'])}",
        f"- CDSR-only missing keys: {len(compatibility['cdsr_missing_keys'])}",
        f"- additional parameters: {compatibility['additional_parameters']}",
        f"- compatibility audit: **{'PASS' if compatibility['pass'] else 'FAIL'}**",
        "",
        "The six CDSR-only missing keys are exactly the six new alpha logits;",
        "the released backbone-pretrained audit has the same missing/unexpected",
        "keys as A0 plus only these six expected scalars.",
        "",
        "## 5. Real BCSS 20-step smoke",
        "",
        f"- parsed training samples: {protocol['parsed_training_samples']}",
        f"- batch / image size: {protocol['batch_size']} / {protocol['image_size']}",
        f"- seed / steps: {protocol['seed']} / {protocol['steps']}",
        f"- precision: {protocol['precision']}",
        f"- official loss weights: {protocol['loss_weights']}",
        f"- optimizer: {protocol['optimizer']}",
        f"- base LR / max-step: {protocol['base_lr']} / {protocol['max_step']}",
        f"- momentum / weight decay: {protocol['momentum']} / {protocol['weight_decay']}",
        f"- PyTorch / CUDA: {protocol['pytorch']} / {protocol['cuda_runtime']}",
        f"- GPU: {protocol['gpu']}",
        f"- step-1 / step-20 loss: {records[0]['loss']:.6f} / {records[-1]['loss']:.6f}",
        f"- minimum observed loss: {min(item['loss'] for item in records):.6f}",
        f"- all outputs/losses/alpha gradients finite: **{smoke['all_finite']}**",
        "",
        "Step-20 mechanism state:",
        "",
        "| Stage | N mean | N std | G_sem mean | G_ctx mean | gamma_sem | gamma_ctx | finite |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for stage in ("stage1", "stage2", "stage3"):
        value = last_diagnostics[stage]
        lines.append(
            f"| {stage} | {value['need']['mean']:.4f} | "
            f"{value['need']['std']:.4f} | "
            f"{value['semantic_gate']['mean']:.4f} | "
            f"{value['context_gate']['mean']:.4f} | "
            f"{value['gamma_sem']:.6f} | "
            f"{value['gamma_context']:.6f} | {value['all_finite']} |"
        )

    lines.extend(
        [
            "",
            "## 6. Matched weight-decay-only shadow audit",
            "",
            "A shadow copy of every alpha logit used the same group-2 LR, poly",
            "schedule, momentum, and weight decay, but received zero task gradient.",
            "The preregistered numerical criterion is divergence from the shadow",
            "by at least one float32 ULP at the initial logit. LR equality held at",
            "all 20 steps.",
            "",
            "| Alpha | step-20 task grad | actual logit movement | WD-only movement | task-excess logit | ULP | measurable |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name in sorted(final):
        value = final[name]
        gradient = records[-1]["alphas"][name]["task_gradient"]
        lines.append(
            f"| {name} | {scientific(gradient)} | "
            f"{scientific(value['actual_logit_movement'])} | "
            f"{scientific(value['weight_decay_only_logit_movement'])} | "
            f"{scientific(value['task_excess_logit_movement'])} | "
            f"{scientific(value['float32_logit_ulp_threshold'])} | "
            f"{value['measurable_task_excess']} |"
        )
    lines.extend(
        [
            "",
            f"Result: **{passed_alphas}/6** alpha scalars show measurable",
            "task-excess movement. F28_2 semantic/context gradients are nonzero",
            "mathematically, but their optimizer updates remain below float32",
            "resolution and are erased by rounding at every observed step.",
            f"Even the four measurable scalars are only 1-3 ULPs from their",
            f"shadows; task-excess is {min(passed_excess_ratios):.3f}% to",
            f"{max(passed_excess_ratios):.3f}% of the matched weight-decay-only",
            "movement. This is weak partial activation, not a healthy readiness",
            "pass.",
            "",
            "## 7. Resource profile",
            "",
            "Measured on the same RTX 5090 with batch20, 224x224, BF16, three",
            "warmups and ten synchronized measured iterations per mode.",
            "",
            "| Quantity | CDSR vs A0 | frozen budget | result |",
            "|---|---:|---:|---|",
            f"| parameters | +{deltas['additional_parameters']} "
            f"({percent(deltas['parameter_percent'])}) | exactly +6 | PASS |",
            f"| estimated FLOPs | {percent(deltas['estimated_flops_percent'])} | <+0.1% | PASS |",
            f"| forward median latency | {percent(deltas['forward_median_latency_percent'])} | <+5% | PASS |",
            f"| train median latency | {percent(deltas['train_median_latency_percent'])} | <+10% | PASS |",
            f"| forward peak memory | {percent(deltas['forward_peak_memory_percent'])} | reported | — |",
            f"| train peak memory | {percent(deltas['train_peak_memory_percent'])} | reported | — |",
            "",
            "FLOPs combine exact Conv2d/Linear multiply-add counts with the",
            "explicit analytical-operation estimate recorded in the profile JSON.",
            "",
            "## 8. Readiness matrix",
            "",
            "| Check | Result |",
            "|---|---|",
            "| frozen formula unchanged | PASS |",
            "| A0 uniform compatibility | PASS |",
            "| six alpha scalars only | PASS |",
            "| all local/server tests | PASS |",
            "| batch20 BF16 finite | PASS |",
            "| pretrained audit | PASS |",
            "| optimizer coverage | PASS |",
            "| matched shadow LR | PASS |",
            "| resource budget | PASS |",
            "| measurable task-excess for all six alphas | **FAIL (4/6)** |",
            "| overall readiness | **FAIL** |",
            "",
            "## 9. Stop decision",
            "",
            "Per the frozen specification, engineering readiness is not granted.",
            "Do not start the 25-epoch BCSS experiment. No architecture or",
            "hyperparameter remedy is proposed or applied in this branch; the",
            "result is stopped for review exactly at the readiness gate.",
            "",
            "## 10. Reproduction commands",
            "",
            "```bash",
            "python -m pytest -q",
            "python tools/check_cdsr_a0_compatibility.py --checkpoint <A0.pth> \\",
            "  --output-json audit/results/cdsr_a0_compatibility.json",
            "python tools/smoke_cdsr.py --train-root <BCSS-training> \\",
            "  --weights <ResNet38.params> --dataset bcss --batch-size 20 \\",
            "  --steps 20 --formal-epochs 25 --image-size 224 --seed 42 \\",
            "  --output-json audit/results/cdsr_readiness_smoke.json",
            "python tools/profile_cdsr.py --batch-size 20 --image-size 224 \\",
            "  --warmup 3 --iterations 10 \\",
            "  --output-json audit/results/cdsr_resource_profile.json",
            "```",
            "",
            "CDSR_IMPLEMENTATION_READINESS_FAIL",
        ]
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
