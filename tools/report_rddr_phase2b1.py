"""Render the complete report from verified immutable CSV/JSON, no inference."""
import argparse
import csv
import json
from pathlib import Path
import math


def rows(root,name):
    with (root/f"rddr_phase2b1_{name}.csv").open(newline="",encoding="utf-8") as f:
        result=list(csv.DictReader(f))
    for row in result:
        for k,v in row.items():
            if v in ("True","False"): row[k]=v=="True"
            else:
                try: row[k]=float(v)
                except ValueError: pass
    return result


def table(items,columns):
    def fmt(value):
        if isinstance(value,float):
            if not math.isfinite(value): return "NA"
            if value.is_integer(): return str(int(value))
            return f"{value:.6f}"
        return str(value)
    return "| "+" | ".join(label for _,label in columns)+" |\n|"+"---|"*len(columns)+"\n"+"\n".join(
        "| "+" | ".join(fmt(row.get(key,"")) for key,_ in columns)+" |" for row in items)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--report-dir",required=True)
    p.add_argument("--output",help="Optional unique report path; never overwrite")
    args=p.parse_args(); root=Path(args.report_dir)
    output=Path(args.output) if args.output else root/"rddr_phase2b1_dual_hypothesis_context_adjudication_report.md"
    if output.exists(): raise FileExistsError(output)
    s=json.loads((root/"rddr_phase2b1_summary.json").read_text(encoding="utf-8"))
    runtime=json.loads((root/"rddr_phase2b1_runtime.json").read_text(encoding="utf-8"))
    manifest=json.loads((root/"rddr_phase2b1_population_manifest.json").read_text(encoding="utf-8"))
    verify=json.loads((root/"rddr_phase2b1_independent_verification.json").read_text(encoding="utf-8"))
    assert verify["status"]=="PASS" and verify["decision"]==s["decision"]
    groups=rows(root,"all_groups"); metrics=rows(root,"anchor_metrics")
    support=rows(root,"support_diagnostics"); signs=rows(root,"sign_decision")
    adjud=rows(root,"adjudication"); safety=rows(root,"deep_wrong_safety")
    bygroup={r["group"]:r for r in groups}
    am={(r["group"],r["estimator"]):r for r in metrics}
    ci=s["ci"]; a=ci["adjudication_image_auroc"]; b=s["sign_primary"]
    ca,cm=ci["anchor_fixed_accuracy_delta"],ci["anchor_fixed_miou_delta"]
    dw,tdw=ci["Deep_Wrong_anchor_fixed_accuracy_delta"],ci["Top20_Deep_Wrong_anchor_fixed_accuracy_delta"]
    general=[("group","Group"),("targets","FG targets"),("image_balanced_auroc","Image AUROC"),
             ("sign_balanced_accuracy","Sign BA"),("fixed_average_accuracy","Fixed acc"),("anchor_accuracy","Anchor acc"),
             ("anchor_fixed_accuracy_delta","Acc delta"),("anchor_fixed_miou_delta","mIoU delta")]
    utility=[("estimator","Estimator"),("accuracy","Accuracy"),("miou","mIoU"),("dice","Dice"),("nll","NLL"),("brier","Brier")]
    signcols=[("group","Group"),("targets","Winner targets"),("accuracy","Sign accuracy"),("balanced_accuracy","Pooled BA"),
              ("macro_f1","Macro F1"),("deep_win_recall","Deep-Win recall"),("shallow_win_recall","Shallow-Win recall")]
    safetycols=[("group","Group"),("targets","Targets"),("shallow_accuracy","Shallow"),("deep_accuracy","Deep"),
                ("fixed_average_accuracy","FixedAvg"),("anchor_accuracy","Anchor"),("anchor_fixed_accuracy_delta","Anchor-Fixed"),
                ("anchor_shallow_delta","Anchor-Shallow")]
    def safety_rows(name):
        return [dict(r,anchor_shallow_delta=r["anchor_accuracy"]-r["shallow_accuracy"]) for r in rows(root,name)]
    sections=["# RDDR Phase-2B1 Dual-Hypothesis Context Adjudication Audit\n",
        f"**最终判定：`{s['decision']}`。** Gate A/B/C/D = "+" / ".join("PASS" if s["gates"][k] else "FAIL" for k in "ABCD")+"。\n\n"
        f"Delta 有排序信号：image AUROC={a['observed']:.6f}。但固定 sign 的 Deep-Win recall={100*b['deep_win_recall']:.4f}%，"
        f"BA={100*b['balanced_accuracy']:.4f}%；anchor 相对 FixedAvg 的 accuracy={100*ca['observed']:+.4f}pp，mIoU={100*cm['observed']:+.4f}pp。"
        "不能将排序能力、accuracy提升或安全改善替代方向召回率及mIoU门槛。\n\n"
        "所有表中比例默认0–1，pp=百分点；NLL/Brier不是百分比。统计基于原生28-grid，**不是官方224/TTA final-CAM指标**。\n",
        "## 1. Provenance / frozen evidence\n\n"
        f"- Pure A0：`{runtime['a0_commit']}`\n- Extraction commit：`{s['extraction_commit']}`\n"
        f"- Analysis commit：`{s['analysis_commit']}`\n- Checkpoint：`{runtime['checkpoint']}`\n"
        f"- Checkpoint SHA256：`{s['checkpoint_sha256']}`\n- Native observation SHA256：`{s['native_observations_sha256']}`\n"
        f"- Sufficient statistics SHA256：`{runtime['sufficient_statistics_sha256']}`\n\n"
        "Phase0已证明冲突信号存在；Phase1 feature disposal、Phase2A receiver suppression和Phase2B0固定deep-anchor路线均未通过各自冻结门槛。"
        "本轮从A0独立开始，不继承旧模型改造；不删除旧实验，不改变此前NOGO。\n",
        "## 2. Exact commands / environment / resources\n\n```bash\ncd "+runtime["working_directory"]+"\n"
        +runtime["command"]+"\n"+s["exact_analysis_command"]+"\n```\n\n"
        f"{runtime['gpu']}；Python {runtime['python']} / PyTorch {runtime['torch']} / NumPy {runtime['numpy']}。"
        "batch1，BF16前向、FP32 softmax/support。benchmark=False，matmul=none，conv=tf32，与冻结Phase0 backend一致。\n\n"
        f"提取总耗时{runtime['total_seconds']:.2f}s（含观测压缩落盘），forward/support={runtime['forward_support_seconds']:.2f}s；"
        f"离线统计和10000 bootstrap={runtime['analysis_seconds']:.2f}s。CUDA峰值allocated={runtime['peak_cuda_allocated_bytes']/2**30:.3f}GiB，"
        f"reserved={runtime['peak_cuda_reserved_bytes']/2**30:.3f}GiB；原生概率观测缓存={runtime['native_observations_bytes']/2**20:.2f}MiB。"
        "缓存不是模型checkpoint，不保存全数据集pair张量。\n",
        "## 3. Tensor / preprocessing / GT contract\n\n"
        "未修改的A0 Net.forward加只读hook。F28_raw=[1,512,28,28]、Ddeep=[1,4096,28,28]，"
        "Ls=ic1(F28_raw)、Ld=fc8(Ddeep)，ps/pd=softmax(logits.float())。不对logits施加ReLU、CAM归一化、presence或TTA；"
        "原网络内部激活不改。图像224/bilinear/ImageNet normalization，eval、requires_grad=False、no_grad。\n\n"
        "GT及历史mask从224 nearest到28；只在metric target中保留0–3，4/255不计入metric。"
        "GT不进入support、anchor或context；包括GT背景位置在内的所有合法邻居都参与无监督support。没有background预测自动修正。\n",
        "## 4. Conflict / frozen groups\n\n"
        "q=clip(JS(ps,pd)/ln2,0,1)。Hard disagreement=argmax(ps)!=argmax(pd)。Top20严格复用历史mask，不重新选择。\n\n"
        +table([dict(group=g,full=v,native=manifest["projected_counts"][g]) for g,v in manifest["full_resolution_counts"].items()],
               [("group","Group"),("full","224 count"),("native","28 count")])
        +f"\n全部3418缓存SHA及逐图历史人数通过；重新提取native q最大误差={runtime['frozen_q_max_abs_difference']}。"
        "原始历史逐像素文件没有保留，本轮复用Phase2A按原代码/backend重建、逐图人数exact的immutable cache；不声称与不存在的原始像素哈希比较。\n",
        "## 5. Neighborhood\n\n"
        "固定15×15/radius7，仅图内邻居，排除self。角落63、中部224个source。无距离权重、Top-k或邻居softmax。"
        "支持度用合法邻居数除法，不把padding计入分母。\n",
        "## 6. Two full-distribution hypotheses\n\n"
        "hS=ps(i)，hD=pd(i)，e_j=ps(j)。四类完整概率分布；不one-hot，不预设deep正确，不乘source/target reliability。\n",
        "## 7. Support equations and measurements\n\n"
        "```text\ncS_ij=clip(1-JS(ps_i,ps_j)/ln2,0,1)\ncD_ij=clip(1-JS(pd_i,ps_j)/ln2,0,1)\n"
        "SS_i=mean_j(cS_ij); SD_i=mean_j(cD_ij)\n```\n\n"
        "自然对数、epsilon=1e-8，epsilon位置与Phase0逐项一致；temperature=1。\n\n"
        +table([r for r in support if r["group"] in ("all","Deep_Win","Shallow_Win","Both_Wrong")],
               [("group","Group"),("SS_pixel_mean","Mean SS"),("SD_pixel_mean","Mean SD"),("Delta_pixel_mean","Mean Delta"),("wD_pixel_mean","Mean wD")]),
        "## 8. Primary Delta and sign\n\n"
        "Delta=SD-SS。**Delta>0选deep；否则选shallow，包括Delta=0。** 不调阈值、不改变符号。"
        f"Eligible winner population中零值tie={b['exact_zero_ties']}。"
        f"离线重算wd/anchor/FixedAvg相对CUDA原生观测误差：{s['numerical_replay']}。主指标使用保存的CUDA预测概率。\n",
        "## 9. Adjudication population\n\n"
        f"Foreground={s['foreground_targets']:,}；hard disagreement={s['hard_disagreement_targets']:,}；"
        f"exactly-one-correct conflict={s['adjudication_targets']:,}。"
        "只有最后一组形成二分类裁决目标。Agreement、both-wrong不会被悄悄作为Shallow-Win负例。\n",
        "## 10. Deep-Win / Shallow-Win prevalence\n\n"
        f"Deep-Win={s['deep_win_count']:,}，Shallow-Win={s['shallow_win_count']:,}；"
        f"Deep-Win prevalence={s['deep_win_count']/s['adjudication_targets']:.6f}。"
        "Y=1表示deep正确/shallow错误，Y=0相反。GT仅构建审计标签，不提供给support。\n",
        "## 11. Exact pooled / image-balanced AUROC and AP\n\n"
        +table([r for r in adjud if r["group"] in ("all","Top20","hard_disagreement")],
               [("group","Group"),("auroc","Pooled AUROC"),("image_balanced_auroc","Image AUROC"),("auprc","Pooled AP"),
                ("image_balanced_auprc","Image AP"),("prevalence","Prevalence"),("auroc_eligible_images","AUC images")])
        +"\n使用精确排序和tie处理，不使用4096bin近似。AUROC缺少任一标签则NA；AP无正例为NA，全正例为1。"
        "Primary是image-balanced AUROC，不能结果出来后换成pooled。\n",
        "## 12. Primary AUROC confidence interval\n\n"
        +table([a],[("observed","AUROC"),("ci95_low","95% low"),("ci95_high","95% high"),("eligible_images","Eligible images")])
        +"\n在全部3418图上重采样；每次忽略该指标未定义的图像。无正负两类图像没有人为赋AUROC=0.5。\n",
        "## 13. Fixed sign decision accuracy / BA / F1\n\n"
        +table([r for r in signs if r["group"] in ("all","Top20","Bottom80")],signcols)
        +f"\n主BA={b['balanced_accuracy']:.6f}，95%CI=[{ci['sign_balanced_accuracy']['ci95_low']:.6f},{ci['sign_balanced_accuracy']['ci95_high']:.6f}]。"
        f"辅助image-balanced BA={b['image_balanced_accuracy']:.6f}；按确认合同，不能替代主pooled BA。\n",
        "## 14. Winner recalls / strength diagnostic\n\n"
        f"Deep-Win recall={100*b['deep_win_recall']:.4f}%（门槛55%）；Shallow-Win recall={100*b['shallow_win_recall']:.4f}%（门槛55%）。"
        "Gate B要求BA和双方recall同时满足，不以总体accuracy或AUC替代。\n\n"
        f"Strength=abs(Delta)，在全部FG一次冻结higher-quantile边界：{s['strength_quintile_edges']}。\n\n"
        +table([r for r in signs if str(r["group"]).startswith("Strength")],signcols)
        +"\nStrength只观察margin与正确性的关系，不选择运行子集或修改权重。\n",
        "## 15. Contextual anchor\n\n"
        "```text\nwD=SD/(SS+SD+eps); wS=1-wD\np_anchor=wS*ps+wD*pd\np_fixed=.5*ps+.5*pd\n```\n\n"
        "无q-dependent beta、learned gate、temperature或搜索。Context-only为mean_j(ps_j)，不参与anchor公式。\n",
        "## 16. Foreground semantic utility\n\n"
        +table([r for r in metrics if r["group"]=="all"],utility)
        +f"\nAnchor-FixedAvg accuracy={100*ca['observed']:+.4f}pp，mIoU={100*cm['observed']:+.4f}pp。"
        "四类mIoU/macroDice从总confusion计算，zero-union/denominator类别为NA并排除，不设为完美。"
        "NLL=-log(pGT+eps)，Brier为四类直接平方误差之和再平均。\n\n"
        +table([dict(class_id=k,fixed=am['all','fixed_average'][f'class{k}_iou'],anchor=am['all','anchor'][f'class{k}_iou'],
                     delta_pp=100*(am['all','anchor'][f'class{k}_iou']-am['all','fixed_average'][f'class{k}_iou'])) for k in range(4)],
               [("class_id","Class"),("fixed","Fixed IoU"),("anchor","Anchor IoU"),("delta_pp","Delta pp")])
        +"\n原生28-grid weak-logit诊断不能直接解释为官方final-CAM复现增益。\n",
        "## 17. Conflict-only utility\n\n"+table([bygroup[g] for g in ("hard_disagreement","adjudication")],general),
        "## 18. Frozen Top20 / Bottom80\n\n"+table(rows(root,"top20_bottom80"),general)
        +f"\nTop20 anchor-fixed accuracy delta={100*ci['Top20_anchor_fixed_accuracy_delta']['observed']:+.4f}pp。"
        "Top20 nearest投影后不强制20%，没有按本轮Delta重新取Top20。\n",
        "## 19. Deep-Correct / Deep-Wrong and hard safety\n\n"+table(safety_rows("deep_strata"),safetycols)
        +f"\nGlobal Deep-Wrong delta={100*dw['observed']:+.4f}pp；Top20 Deep-Wrong={100*tdw['observed']:+.4f}pp。\n\n"
        +table(safety,[("stratum","Deep-Wrong stratum"),("targets","Targets"),("fixed_average_accuracy","Fixed"),
                      ("anchor_accuracy","Anchor"),("delta","Delta"),("hard_line_failed","<=-10pp")])
        +"\n所有15个分层提前固定，下降参照均为FixedAvg。检查汇总分层，不按单图/像素或事后子集触发。"
        "D还要求全体>=-2pp、Top20>=-3pp。空分层为NA并报告覆盖率，不伪造通过值。\n",
        "## 20. Symmetric shallow-correct / shallow-wrong audit\n\n"+table(safety_rows("shallow_strata"),safetycols)
        +"\n安全改善不等于没有shallow bias；同时观察浅层正确与浅层错误的代价。\n",
        "## 21. Both-wrong / third-class recovery\n\n"+table([r for r in metrics if r["group"]=="Both_Wrong"],utility)
        +"\n"+table([r for r in rows(root,"echo") if r["group"]=="Both_Wrong"],
               [("group","Group"),("targets","Targets"),("anchor_differs_both_count","Different from both"),("anchor_correct_third_class_count","Correct third class")])
        +"\n该纠正仅secondary signal，不改变任何门槛。\n",
        "## 22. Context consensus diagnostic\n\n"
        +table([r for r in support if r["group"] in ("all","Deep_Win","Shallow_Win","Both_Wrong","Top20")],
               [("group","Group"),("JS_ctx_shallow_pixel_mean","JS(ctx,ps)"),("JS_ctx_deep_pixel_mean","JS(ctx,pd)"),
                ("consensus_closer_deep_fraction","Fraction closer to deep")])
        +"\nJS(mean_j ps_j,hypothesis)与mean_j JS(ps_j,hypothesis)并不相等；共识只是解释性参照，未替换primary mean support。\n",
        "## 23. Frozen HFRM transition groups\n\n"+table(rows(root,"hfrm_groups"),
               [("group","Group"),("Delta_pixel_mean","Mean Delta"),("SS_pixel_mean","Mean SS"),("SD_pixel_mean","Mean SD"),
                ("fixed_average_accuracy","Fixed acc"),("anchor_accuracy","Anchor acc"),("anchor_fixed_accuracy_delta","Delta acc")])
        +"\n历史by-CH命名实际是raw→full HFRM transition，包括semantic veto+context，**不是CH-only因果干预**。\n",
        "## 24. Frozen q quintiles\n\n"+f"沿用并校验Phase2B0 native q边界：{s['q_quintile_edges']}。tie归较低分组。\n\n"
        +table(rows(root,"quintiles"),[("group","Bin"),("hard_disagreement_prevalence","Hard prevalence"),
               ("positive","Deep-Win n"),("negative","Shallow-Win n"),("image_balanced_auroc","Image AUC"),
               ("sign_balanced_accuracy","Sign BA"),("anchor_fixed_accuracy_delta","Accuracy delta"),("anchor_fixed_miou_delta","mIoU delta")]),
        "## 25. Boundary / interior\n\n"+table(rows(root,"boundary_interior"),general)
        +"\n"+table(rows(root,"boundary_interior"),[("group","Group"),("deep_wrong_targets","Deep-Wrong n"),("deep_wrong_anchor_fixed_accuracy_delta","Deep-Wrong delta")])
        +"\n沿用fullres FG-FG 8-neighbor transition欧氏距离<=7px的边界，先224构造再nearest投影；未进入support计算。\n",
        "## 26. Per-class adjudication\n\n"+table(rows(root,"per_class"),[("group","Class"),("positive","Deep-Win n"),
               ("negative","Shallow-Win n"),("auroc","Pooled AUC"),("image_balanced_auroc","Image AUC"),
               ("sign_balanced_accuracy","Sign BA"),("fixed_average_accuracy","Fixed acc"),("anchor_accuracy","Anchor acc"),("anchor_fixed_accuracy_delta","Delta")]),
        "## 27. Support calibration\n\n"+table(rows(root,"calibration"),[("low","Lower inclusive"),("high","Upper"),("upper_inclusive","Upper inclusive"),
               ("targets","Winner n"),("mean_wD","Mean wD"),("empirical_deep_win_probability","P(Deep-Win)")])
        +"\n固定五桶，空桶NA。wD是support ratio，不自动等价于校准后的P(deep正确)。无温度、偏移或分桶后重校准。\n",
        "## 28. Echo diagnostics\n\n"+table([r for r in rows(root,"echo") if r["group"] in ("all","Top20","hard_disagreement","Deep_Correct","Deep_Wrong")],
               [("group","Group"),("anchor_equals_deep","Anchor=deep"),("anchor_equals_shallow","Anchor=shallow"),("anchor_differs_both_count","Neither n")])
        +"\nshallow/deep相同的像素同时计入两列，故全体两列不应强制相加为1；hard disagreement最能区分echo倾向。\n",
        "## 29. Winner oracle reference\n\n"
        f"Exactly-one-correct conflict上的oracle winner accuracy={s['oracle_winner_accuracy']:.6f}；"
        f"固定sign实际accuracy={b['accuracy']:.6f}，距oracle={100*s['adjudication_gap_to_oracle']:.4f}pp。"
        "Oracle只用于诊断，不调公式、不向anchor传递GT。\n",
        "## 30. Paired bootstrap / independent verification\n\n"+table(rows(root,"bootstrap"),
               [("metric","Metric"),("observed","Observed"),("ci95_low","95% low"),("ci95_high","95% high"),("eligible_images","Eligible images")])
        +"\n10000次、seed42、相同图像索引配对；AUROC按image mean，BA重加2×2confusion，accuracy/mIoU重加4×4confusion，"
        "没有target/pair-level naive bootstrap。所有replicate均保留。\n\n"
        f"独立NumPy/SciPy验证：{verify['status']}；3418图AUROC用rankdata单独重算，"
        f"{verify['independent_real_support_points']}个固定真实位置用显式邻居重算support（max error={verify['max_support_error']:.9g}）；"
        "另复算32个完整bootstrap replicate、全部10000次CI分位点和15个安全分层。"
        "21项单元测试日志及独立verification.json随交付保存。\n",
        "## 31. Frozen Gate A/B/C/D\n\n"+table([dict(gate=k,result="PASS" if s['gates'][k] else "FAIL") for k in "ABCD"],[("gate","Gate"),("result","Result")])
        +"\nA: imageAUC>=.65且CI下界>.5。B: pooledBA>=.60且双方recall>=.55。"
        "C: accuracy、mIoU都提升，且至少一项差值CI下界>0。D:全体/Top20 Deep-Wrong满足-2/-3pp容忍，所有预注册安全分层不触及-10pp。"
        "BA的CI跨0.60不会替代其point门槛，更不能替代Deep-Win recall门槛。\n",
        "## 32. STRONG_SIGNAL\n\n"
        f"STRONG_SIGNAL = {str(s['strong_signal']).upper()}。需要image AUROC>=.70、anchor-fixed mIoU>=+1pp、全体Deep-Wrong delta>=0同时满足。"
        "它是辅助标志，不能覆盖硬门槛。\n",
        "## 33. Scientific interpretation / boundaries\n\n"
        "本轮应分清三个问题：Delta能否排序winner、固定零阈值能否给出可靠双向判断、支持比值融合能否提升语义指标。"
        f"本次AUC={a['observed']:.4f}，支持局部上下文包含winner排序信息；但Deep-Win recall={100*b['deep_win_recall']:.2f}%，"
        "固定sign明显偏向shallow。AUROC对单调平移不敏感，因此高AUC本身不保证零阈值决策有效；这里仅解释差异，不执行任何平移/调参。\n\n"
        f"Anchor准确率提高{100*ca['observed']:.4f}pp而macro mIoU变化{100*cm['observed']:+.4f}pp，不能宣称整体分割质量提升。"
        f"全局class3 IoU差为{100*(am['all','anchor']['class3_iou']-am['all','fixed_average']['class3_iou']):+.4f}pp，"
        "类别分布与macro/pixel加权的不同解释了两种指标不能互相替代；不据此新增class规则。"
        "Top20/Bottom80和q分位组分别重算的mIoU均可能提升，而合并后mIoU下降：各组mIoU不能按样本量平均还原整体mIoU，"
        "每类union及其权重也随预测改变。完整confusion可逐项相加，分层/总体数字已经独立复核，不用最佳分组替代全体结果。\n\n"
        f"排序信号并非跨类普适：class2/class3的image AUROC分别为{bygroup['class2']['image_balanced_auroc']:.4f}/"
        f"{bygroup['class3']['image_balanced_auroc']:.4f}，均低于0.5；class3的Shallow-Win仅{int(bygroup['class3']['negative'])}个，"
        "需保留不平衡背景，不能只引用整体AUC。Strength最大一组也未呈现更高sign正确率，abs(Delta)不等于已校准置信度。\n\n"
        "还需区别hard sign和soft anchor：前者偏向shallow，后者仍可能更多echo deep，因为anchor融合完整概率分布而非按sign硬切换，"
        "两者预测不必一致。这里观察两者局限，不进行阈值平移或重新校准。\n\n"
        f"Deep-Wrong安全明显改善（全体{100*dw['observed']:+.4f}pp，Top20 {100*tdw['observed']:+.4f}pp），但这只说明"
        "相对FixedAvg没有重现上轮的deep-following灾难，不能抵消B/C失败。所有class/Top20/Strength/consensus/oracle结果仅解释，"
        "不替换primary，不搜索阈值/温度/窗口/权重，不追加训练。\n\n"
        "本轮按研究项目实施交付流程保存冻结合同、独立A0分支、可运行命令、测试、CSV/JSON及完整报告；"
        "旧实验及checkpoint不删除不覆盖。\n",
        "## 34. Exact decision / STOP\n\n"
        "按用户在看结果前批准的优先级：D失败优先UNSAFE；否则A或B失败为NOGO；否则C失败为FUSION_UTILITY_FAIL；全过才GO。"
        "本次A通过、B失败、C失败、D通过，因此使用预先补齐的B失败规则，而非事后选择标签。"
        "**不启动Phase2B2训练，不访问test/LUAD或其他seed。报告与PR交付后停止。**\n\n"
        +"DECISION = "+s["decision"]]
    output.write_text("\n".join(sections)+"\n",encoding="utf-8")
    print(str(output))


if __name__=="__main__": main()
