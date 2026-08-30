"""Render the complete frozen Phase-2B1.5 report after independent verification."""
import argparse
import csv
import json
import math
from pathlib import Path

P="rddr_phase2b15_"


def rows(root,name):
    with (root/(P+name+".csv")).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def fmt(x):
    if x is None:return "NA"
    if isinstance(x,bool):return str(x).upper()
    try:
        f=float(x)
        if not math.isfinite(f):return "NA"
        return str(int(f)) if f==int(f) else f"{f:.6f}"
    except (ValueError,TypeError):return str(x)


def table(data,columns):
    text=["| "+" | ".join(label for label,key in columns)+" |","|"+"---|"*len(columns)]
    text += ["| "+" | ".join(fmt(row.get(key)) for label,key in columns)+" |" for row in data]
    return "\n".join(text)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--report-dir",required=True)
    p.add_argument("--output",required=True);args=p.parse_args();root=Path(args.report_dir);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    s=json.loads((root/(P+"summary.json")).read_text());rt=json.loads((root/(P+"runtime.json")).read_text())
    v=json.loads((root/(P+"verification.json")).read_text());assert v["status"]=="PASS" and v["decision"]==s["decision"]
    allrows=rows(root,"all_groups");ag={r["group"]:r for r in allrows}
    adj=rows(root,"symmetric_adjudication");ad={(r["group"],r["score"]):r for r in adj}
    anchor=rows(root,"symmetric_anchor");am={(r["group"],r["estimator"]):r for r in anchor}
    bias=rows(root,"same_family_bias");bm={(r["group"],r["field"]):r for r in bias}
    ci=s["ci"];cirows=[r for r in rows(root,"bootstrap") if "__" not in r["metric"]]
    source=rows(root,"source_branch_reversal");contexts=rows(root,"context_sources")
    mass=rows(root,"candidate_mass");gt=rows(root,"gt_context_availability");rc=rows(root,"class23_root_cause")
    bc=[("Group","group"),("B_S","B_S_mean"),("B_D","B_D_mean"),("B_family","B_family_mean")]
    ac=[("Score","score"),("Mean score","mean_score"),("Pooled AUC","auroc"),("Image AUC","image_auroc"),
        ("BA","balanced_accuracy"),("Deep-Win recall","deep_win_recall"),("Shallow-Win recall","shallow_win_recall")]
    mc=[("Estimator","estimator"),("Accuracy","accuracy"),("mIoU","miou"),("Dice","dice"),("NLL","nll"),("Brier","brier")]
    parts=["# RDDR Phase-2B1.5 Adjudication Bias Decomposition & Third-Evidence Audit\n",
           "**最终判定：`"+s["decision"]+"`。**\n\n"
           "同分支来源偏差得到支持；对称化明显改善裁决，第三证据探针也满足预注册门槛。"
           "但class2仅勉强达到宽松门槛、class3证据不足，不能宣称所有类别反转已解决。"
           "这是零训练机制审计，不是新模型的官方mIoU提升。\n\n"
           "除注明百分比/pp外，表中均使用0–1比例；pp为百分点。所有语义指标为原生28-grid，"
           "不是官方224/TTA/final-CAM指标。Phase-2B1 NOGO保持不变。\n"]
    def section(i,title,text):parts.append(f"## {i}. {title}\n\n{text}\n")
    section(1,"Provenance / immutable inputs",
        f"- Pure A0：`{rt['a0_commit']}`\n- Probe commit：`{rt['code_commit']}`\n- Analysis commit：`{rt['analysis_commit']}`\n"
        f"- Checkpoint：`{rt['checkpoint']}`\n- Checkpoint SHA256：`{rt['checkpoint_sha256']}`\n"
        f"- Native cache：`{rt['native']}`\n- Cache SHA256：`{rt['native_sha256']}`\n"
        f"- Derived observations SHA256：`{rt['derived_sha256']}`\n- Statistics SHA256：`{rt['statistics_sha256']}`\n\n"
        "全部3418 validation图像、C0 Full25 seed42。仅复用缓存中的ps/pd及冻结GT/分组；"
        "未加载模型、未前向、未创建optimizer、未反传、未写checkpoint，未访问test/LUAD/其他seed。"
        "旧报告、缓存、baseline均未删除或覆盖。")
    section(2,"Exact commands / environment / resources",
        "```bash\ncd /home/duyanhong/DZWdeRepo-rddr-phase2b15\n"+rt["command"]+"\n"
        +"OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "+rt["analysis_command"]+"\n```\n\n"
        f"{rt['gpu']}；Python {rt['python']} / PyTorch {rt['torch']} / NumPy {rt['numpy']}。"
        "缓存来自冻结BF16前向，本轮以batch1/FP32计算概率邻域，不重新softmax或调整temperature。\n\n"
        f"旧分数校验阶段 {rt['parity_seconds']:.2f}s；新探针 {rt['probe_seconds']:.2f}s；含校验/压缩总计 {rt['total_seconds']:.2f}s；"
        f"全量统计和10,000 bootstrap {rt['analysis_seconds']:.2f}s。"
        f"CUDA allocated峰值 {rt['peak_cuda_allocated_bytes']/2**20:.2f}MiB，reserved {rt['peak_cuda_reserved_bytes']/2**20:.2f}MiB。"
        f"衍生观测 {rt['derived_bytes']/2**20:.2f}MiB，不保存全数据集pair张量。")
    section(3,"Phase-2B1 exact reproduction / populations",
        table([dict(field=k,max_abs=z) for k,z in rt["parity_max_abs"].items()],[("Field","field"),("Maximum absolute difference","max_abs")])+
        "\n先完成全量旧分数校验，再计算新探针；严格容忍上限1e-7未放宽。"
        "原生q、Top20、boundary和Q1–Q5直接复用，不重新挑选。\n\n"+
        table([ag[g] for g in ("all","hard_disagreement","adjudication","Deep_Win","Shallow_Win","Both_Wrong","Top20","Bottom80")],
              [("Population","group"),("Targets","targets")])+
        "\nGT类别0–3为metric targets；背景4/ignore255不计分，但仍可作为无监督support的邻居位置。"
        "旧by_CH标签表示raw→完整HFRM而非CH-only因果干预；历史缓存逐图人数已核验，但不声称拥有不存在的原始历史像素哈希。")
    section(4,"Four-way support definitions",
        "```text\nT_ab(i)=mean_j clip(1-JS(p_a(i),p_b(j))/ln2,0,1)\n"
        "T_SS=S<-S; T_SD=S<-D; T_DS=D<-S; T_DD=D<-D\n```\n\n"
        "natural log、epsilon1e-8、temperature1；15×15/r7、排除self、仅图内邻居。"
        "完整四类概率，不one-hot，不加入source reliability/q weighting/边界项/距离权重。"
        "计算函数只接受ps/pd，GT诊断与它隔离。")
    section(5,"T_SS / T_SD / T_DS / T_DD statistics",
        table([r for r in rows(root,"support_matrix") if r["group"]=="all"],
              [("Support","field"),("Mean","mean"),("Std","std"),("p05","p05"),("p25","p25"),("Median/p50","median"),("p75","p75"),("p95","p95")])+
        "\n全部45个固定分组的同类统计保存在support_matrix.csv。")
    strat=("all","hard_disagreement","Deep_Win","Shallow_Win","Both_Wrong","Top20","Bottom80","class0","class1","class2","class3","boundary","interior")
    btable=[]
    for g in strat:
        row=dict(group=g)
        for key in ("B_S","B_D","B_family"):
            z=bm[g,key];row[key]=f"{float(z['mean']):.6f} [{float(z['ci95_low']):.6f}, {float(z['ci95_high']):.6f}]"
        btable.append(row)
    section(6,"Same-family bias: B_S / B_D / B_family",
        "```text\nB_S=T_SS-T_SD; B_D=T_DD-T_DS\nB_family=.5*(B_S+B_D)\n```\n\n"+
        table([r for r in bias if r["group"]=="all"],[("Bias","field"),("Mean","mean"),("Std","std"),("p05","p05"),("p25","p25"),("Median/p50","median"),("p75","p75"),("p95","p95")])+
        "\n均值及10,000 image-bootstrap 95%CI：\n\n"+table(btable,[("Group","group"),("B_S [CI]","B_S"),("B_D [CI]","B_D"),("B_family [CI]","B_family")])+
        "\n全体两个分支的同源偏好均为正、CI下界均大于0，Gate A通过。这个结论是聚合意义的，不能说每个像素都同向；B_S/B_D的p05仍可为负。")
    section(7,"Source-branch reversal",
        table([r for r in source if r["group"]=="all"],ac)+
        "\n仅更换邻居来源，mean Delta从负转正，Deep-Win recall由26.17%变为88.00%。"
        "这支持来源分支影响绝对支持尺度。deep-source探针即使部分指标更高也不被选作新primary，未搜索阈值。")
    section(8,"Delta_old vs preregistered Delta_sym",
        "```text\nS_S_sym=.5*(T_SS+T_SD); S_D_sym=.5*(T_DS+T_DD)\nDelta_sym=S_D_sym-S_S_sym\n```\n\n"
        "代数上Delta_sym=Delta_old+B_family；B_family是逐位置的双源观测量，不是调出来的常数偏移。"
        "因此不能把分数均值变化与B_family均值当成两份独立证据。\n\n"+
        table([ad["all",k] for k in ("old","sym")],ac))
    section(9,"Zero-point bias shrinkage",
        table([ag[g] for g in ("all","Deep_Win","Shallow_Win")],[("Group","group"),("Old mean","old_mean"),("Old median","old_median"),
              ("Sym mean","sym_mean"),("Sym median","sym_median"),("BiasShrink","bias_shrink")])+
        f"\n全体BiasShrink={ci['BiasShrink']['observed']*100:.2f}%，95%CI [{ci['BiasShrink']['ci95_low']*100:.2f}%, {ci['BiasShrink']['ci95_high']*100:.2f}%]。"
        "这是平均零点偏移缩小的描述，不是‘解释了多少训练性能差距’的因果比例。")
    section(10,"Adjudication AUROC / AP / BA / recalls",
        table([ad["all",k] for k in ("old","sym")],[("Score","score"),("Pooled AUROC","auroc"),("Image AUROC","image_auroc"),("Pooled AP","auprc"),
              ("Image AP","image_auprc"),("Accuracy","accuracy"),("BA","balanced_accuracy"),("Macro F1","macro_f1"),("Deep recall","deep_win_recall"),("Shallow recall","shallow_win_recall"),("AUC images","auc_images")])+
        "\nPrimary只在exactly-one-correct hard conflict上计算。图内单一标签AUROC=NA，不填0.5；无正例AP=NA，全正例AP=1。"
        "所有zero sign固定score>0选deep，其余选shallow。Sym imageAUROC的95%CI为"
        f"[{ci['sym_image_auroc']['ci95_low']:.6f}, {ci['sym_image_auroc']['ci95_high']:.6f}]；"
        f"与old的配对差={ci['sym_minus_old_image_auroc']['observed']:.6f}，CI "
        f"[{ci['sym_minus_old_image_auroc']['ci95_low']:.6f}, {ci['sym_minus_old_image_auroc']['ci95_high']:.6f}]。")
    section(11,"Symmetric anchor utility",
        "```text\nwD_sym=S_D_sym/(S_S_sym+S_D_sym+eps)\np_anchor_sym=(1-wD_sym)*ps+wD_sym*pd\n```\n\n"+
        table([am["all",e] for e in ("fixed_average","anchor_old","anchor_sym")],mc)+
        f"\n相对FixedAvg：accuracy +{ci['anchor_sym_minus_fixed_average_accuracy']['observed']*100:.4f}pp，"
        f"mIoU +{ci['anchor_sym_minus_fixed_average_miou']['observed']*100:.4f}pp，mIoU差值95%CI "
        f"[{ci['anchor_sym_minus_fixed_average_miou']['ci95_low']*100:.4f}, {ci['anchor_sym_minus_fixed_average_miou']['ci95_high']*100:.4f}]pp。"
        f"相对old anchor：mIoU +{ci['anchor_sym_minus_anchor_old_miou']['observed']*100:.4f}pp。\n\n"
        "指标从总体4×4 confusion计算，zero-union类别排除为NA，不做GT背景覆盖。"
        "NLL使用-log(pGT+eps)，Brier为四类平方误差之和。各分组mIoU不能按像素数平均还原总体。")
    section(12,"Deep/shallow safety",
        table(rows(root,"safety_strata"),[("Group","group"),("Targets","targets"),("FixedAvg acc","fixed_average_accuracy"),
              ("Old acc","anchor_old_accuracy"),("Sym acc","anchor_sym_accuracy"),("Sym-Fixed","anchor_sym_minus_fixed_average_accuracy")])+
        "\nDeep-Wrong总体与Top20仍分别高于FixedAvg约4.6814/10.7598pp，但救援幅度小于old anchor；"
        "与此同时Deep-Correct及Shallow-Wrong损失大幅减少。不能只报救援或只报整体提升。"
        "本阶段D门槛针对第三证据的rescue/harm，不偷换为上一轮Gate D。")
    section(13,"Per-class support matrix and semantic IoU",
        table(rows(root,"per_class"),[("Class","group"),("T_SS","T_SS_mean"),("T_SD","T_SD_mean"),("T_DS","T_DS_mean"),("T_DD","T_DD_mean"),
              ("B_S","B_S_mean"),("B_D","B_D_mean"),("B_family","B_family_mean"),("Old mean","old_mean"),("Sym mean","sym_mean"),("Deep-Win","deep_win_count"),("Shallow-Win","shallow_win_count"),("Both-Wrong","both_wrong_count")])+
        "\n各类裁决能力：\n\n"+
        table(rows(root,"per_class"),[("Class","group"),("Old pooled AUC","old_auroc"),("Sym pooled AUC","sym_auroc"),
              ("Old image AUC","old_image_auroc"),("Sym image AUC","sym_image_auroc"),("Old BA","old_balanced_accuracy"),("Sym BA","sym_balanced_accuracy")])+
        "\n下面是**全体混淆矩阵**的各类IoU，不是仅筛选该GT类后得到的macro mIoU：\n\n"+
        table([dict(cls=c,**{e:am["all",e][f"iou_class{c}"] for e in ("fixed_average","anchor_old","anchor_sym")}) for c in range(4)],
              [("Class","cls"),("FixedAvg IoU","fixed_average"),("Old IoU","anchor_old"),("Sym IoU","anchor_sym")]))
    pairrows=[]
    for r0 in rows(root,"ordered_pairs"):
        z=dict(r0);z["auc_images"]=ad[z["group"],"sym"]["auc_images"];z["winner_prevalence"]=ad[z["group"],"sym"]["prevalence"];pairrows.append(z)
    section(14,"All 12 ordered prediction pairs",
        table(pairrows,[("S→D pair","group"),("Targets","targets"),("Deep-Win","deep_win_count"),("Shallow-Win","shallow_win_count"),("Winner prevalence","winner_prevalence"),
              ("Old image AUC","old_image_auroc"),("Sym image AUC","sym_image_auroc"),("AUC images","auc_images"),("Old BA","old_balanced_accuracy"),("Sym BA","sym_balanced_accuracy"),("Support","pair_support")])+
        "\n3→2的Shallow-Win仅73，标记LOW_SUPPORT，不据此下结论。2→3虽通过像素人数门槛，"
        "其image AUROC也仅由1张同时含正负例的图决定；0值不能当成稳健反向信号。"
        "3→0/3→1仅8张可计算AUROC图，保留这一限制，不新增门槛、不选择最佳pair。"
        "全部四项支持度及pooled AUROC也随ordered_pairs.csv保存。")
    section(15,"Class priors / confidence (no calibration)",
        table(rows(root,"class_prior_confidence"),[("GT group","group"),("Head","branch"),("Pred0","predicted_frequency_class0"),("Pred1","predicted_frequency_class1"),
              ("Pred2","predicted_frequency_class2"),("Pred3","predicted_frequency_class3"),("Mean p0","mean_probability_class0"),("Mean p1","mean_probability_class1"),
              ("Mean p2","mean_probability_class2"),("Mean p3","mean_probability_class3"),("Max confidence","mean_max_confidence"),("Entropy nats","entropy_nats")])+
        "\n在GT class3，shallow输出class0/1合计约78.20%，class3仅16.99%；deep输出class3为67.69%。"
        "deep平均max confidence=0.9402，shallow=0.5695，概率几何明显不同。"
        "这些是条件预测分布/置信度差异，不是训练集class prior的因果估计。未进行校准。")
    selected=["Deep_Win","Shallow_Win"]+[f"class{k}" for k in range(4)]
    section(16,"Candidate-class soft mass",
        table([z for z in mass if z["group"] in selected],[("Group (hard only)","group"),("Targets","targets"),("M_s^S","M_s_S"),("M_d^S","M_d_S"),
              ("d-s S margin","margin_S"),("M_s^D","M_s_D"),("M_d^D","M_d_D"),("d-s D margin","margin_D")])+
        "\n所有12个ordered pair的mass和class2/3按winner状态分解保存在candidate_mass.csv。"
        "candidate由ps/pd argmax确定，mass是source概率邻域均值，GT不进入其计算。")
    section(17,"GT contextual availability (audit only)",
        table([z for z in gt if z["group"] in selected],[("Group (hard only)","group"),("Targets","targets"),("Same GT","GT_same_fraction"),
              ("S candidate","GT_shallow_candidate_fraction"),("D candidate","GT_deep_candidate_fraction"),("Other FG","GT_other_fraction"),
              ("Background","GT_background_fraction"),("Ignore","GT_ignore_fraction")])+
        "\n所有比例以图内非self合法邻居为分母。S候选+D候选+otherFG+background+ignore=1；sameGT与这些项重叠，不再相加。"
        "mask nearest到28-grid后计算GT邻域，分数仍完全GT-blind。"
        "GT里有正确类别不等于网络已编码正确语义，只能帮助区分可用性不足与表征/score偏差。")
    rootbase=[z for z in rc if z["group"] in ("class2","class3")]
    rootwinner=("class2_Deep_Win","class2_Shallow_Win","class3_Deep_Win","class3_Shallow_Win")
    section(18,"Dedicated class2/class3 root-cause decomposition",
        table(rootbase,[("Class","group"),("Deep-Win","deep_win_count"),("Shallow-Win","shallow_win_count"),("B_S","B_S_mean"),("B_D","B_D_mean"),
              ("B_family","B_family_mean"),("Old Delta","old_mean"),("Sym Delta","sym_mean"),("Old image AUC","old_image_auroc"),("Sym image AUC","sym_image_auroc"),
              ("Mass/GT targets","mass_gt_targets"),("S mass margin","margin_S"),("D mass margin","margin_D"),("GT same","GT_same_fraction"),("Status","class_evidence_status")])+
        "\n支持度均值覆盖该GT类全体；mass/GT均值只覆盖其中hard disagreement，单独列出人数。\n\n"+
        table([dict(z,**{k:v for q0 in gt if q0["group"]==z["group"] for k,v in q0.items() if k not in ("group","targets")}) for z in mass if z["group"] in rootwinner],
              [("Winner state","group"),("Targets","targets"),("S d-s margin","margin_S"),("D d-s margin","margin_D"),("GT same","GT_same_fraction")])+
        "\n**class2：** image AUROC从0.2755升至0.4514，95%CI [0.4256,0.4771]。仅因预注册门槛是0.45而PASS，"
        "CI仍全部低于0.5，不能写成已恢复正确排序。Deep-Win的GT同类邻居比例84.87%；"
        "Shallow-Win仍有68.37%，但两种source的候选mass都平均偏向错误deep候选，提示方向错误并非仅由GT邻域缺失解释。\n\n"
        "**class3：** B_family=0.3039明显较大，mean Delta从-0.3577移到-0.05377；"
        "image AUROC从0.0717升至0.3409，CI [0.2786,0.4058]。418个Shallow-Win使其按合同UNDERPOWERED，"
        "不是PASS，也不按该门槛计FAIL；低于0.5的观察仍如实保留。Deep-Win处GT同类邻居91.90%，"
        "shallow-source候选margin却为-0.1682，而deep-source为+0.7258，支持语义表征方向差异而非纯邻域缺少正确类别。"
        "Shallow-Win中GT同类比例80.27%，但source mass又可偏错误deep；对称化不是万能校正。\n\n"
        "Class-specific hard-conflict pair组成如下（全部12类均保留）。固定GT类和prediction pair后，winner标签通常天然单一，"
        "其AUROC=NA而非0.5；这张表只解释组成，不从条件pair推导阈值或规则。\n\n"+
        table([z for z in rc if "pair" in z["group"]],[("Class/pair","group"),("Targets","targets"),("Fraction of class hard","hard_pair_fraction"),
              ("Deep-Win","deep_win_count"),("Shallow-Win","shallow_win_count"),("S margin","margin_S"),("D margin","margin_D"),("GT same","GT_same_fraction"),("Support","pair_support")])+
        "\n以上分解区分了family bias、条件预测/置信度偏移、pair组成、context方向错误和统计能力限制；"
        "并未识别各机制的独立因果贡献比例。")
    section(19,"Three context sources and independence",
        "```text\nctx_S=mean_j ps(j); ctx_D=mean_j pd(j)\nctx_sym=.5*(ctx_S+ctx_D)\n```\n\n"+
        table([z for z in contexts if z["group"] in ("all","hard_disagreement","Both_Wrong","Top20_Both_Wrong")],
              [("Group","group"),("Context","context"),("=shallow","equals_shallow"),("=deep","equals_deep"),("Different n","different_from_both"),("Different rate","intrusion_rate")])+
        "\nshallow和deep预测相同时，两列可以重叠。‘第三证据’是提供不同预测/救援的操作性定义，"
        "不是统计独立性证明；ctx本身仍来自同一模型的两组概率。")
    section(20,"Both-Wrong third-class rescue",
        table(rows(root,"both_wrong_rescue"),[("Group","group"),("Context","context"),("Targets","targets"),("Accuracy","accuracy"),("mIoU","miou"),
              ("Different n","different_from_both"),("Correct third n","correct_third_class"),("Rescue rate","rescue_rate"),("Rescue precision","rescue_precision")])+
        f"\nctx_sym Both-Wrong accuracy={ci['ctx_sym_Both_Wrong_accuracy']['observed']*100:.4f}%，CI "
        f"[{ci['ctx_sym_Both_Wrong_accuracy']['ci95_low']*100:.4f}%, {ci['ctx_sym_Both_Wrong_accuracy']['ci95_high']*100:.4f}%]。"
        "由于两个原假设都错，context正确就必然不同于两者，因此accuracy与ThirdClassRescueRate严格恒等，不能算两份独立证据。")
    section(21,"One-correct intrusion / harm",
        table(rows(root,"one_correct_intrusion"),[("Context","context"),("Targets","targets"),("Different n","different_from_both"),
              ("Third wrong n","wrong_third_class"),("Intrusion rate","intrusion_rate"),("Harm rate","harm_rate")])+
        f"\nctx_sym harm={ci['ctx_sym_ThirdClassHarmRate']['observed']*100:.4f}%，CI "
        f"[{ci['ctx_sym_ThirdClassHarmRate']['ci95_low']*100:.4f}%, {ci['ctx_sym_ThirdClassHarmRate']['ci95_high']*100:.4f}%]。"
        "Exactly-one-correct时第三类必错，intrusion与harm严格相等。它不衡量context改选另一个错误原候选造成的全部损失。")
    section(22,"Four semantic states / evidence roles",
        table(rows(root,"three_state_roles"),[("State","group"),("Targets","targets")]+[(e,e+"_accuracy") for e in ("shallow","deep","fixed_average","ctx_S","ctx_D","ctx_sym","anchor_old","anchor_sym")])+
        "\n文件名沿用规格three_state_roles，但实际完整列出四种状态：Both-Correct、Deep-Only、Shallow-Only、Neither-Correct。"
        "Context有救援功能，也可能损害本来正确的状态，不能仅用Neither-Correct证据替代整体评估。")
    section(23,"Context-winner diagnostic",
        "Delta_ctx=JS(ctx,ps)-JS(ctx,pd)，正值更接近deep，零值归shallow。\n\n"+
        table([z for z in rows(root,"context_winner") if z["group"]=="all"],ac)+
        "\nJS(mean context,hypothesis)不等于mean JS(neighbor,hypothesis)；这些是辅助诊断，不替换Delta_sym primary。")
    section(24,"Boundary / interior",
        table(rows(root,"boundary_interior"),bc+[("Old image AUC","old_image_auroc"),("Sym image AUC","sym_image_auroc"),
              ("Old BA","old_balanced_accuracy"),("Sym BA","sym_balanced_accuracy"),("ctx_sym BW rescue","ctx_sym_Both_Wrong_rescue")])+
        "\n直接复用Phase-2B1 frozen boundary：224-grid FG-FG transition的欧氏距离<=7px，再nearest投影到28；不重新定义边界或用于权重。")
    section(25,"Frozen conflict Q1–Q5",
        table(rows(root,"quintiles"),bc+[("Old mean","old_mean"),("Sym mean","sym_mean"),("Old image AUC","old_image_auroc"),
              ("Sym image AUC","sym_image_auroc"),("Old BA","old_balanced_accuracy"),("Sym BA","sym_balanced_accuracy"),("Neither n","both_wrong_count"),("Targets","targets"),("ctx_sym BW rescue","ctx_sym_Both_Wrong_rescue")])+
        "\nNeither-Correct prevalence为表中Neither n/Targets。所有分位边界沿用Phase-2B1，ties归较低组；不按本轮结果重新分箱，Q5好不能单独推进。")
    section(26,"Paired image bootstrap / independent verification",
        table(cirows,[("Metric","metric"),("Observed","observed"),("95% low","ci95_low"),("95% high","ci95_high")])+
        f"\n10,000次、seed42、全部3418图像作为cluster成组重采样；AUROC按image mean，BA/语义指标按confusion重加，bias按sum/count。"
        "45个分组的B_S/B_D/B_family CI另见第6节及CSV；全部159列bootstrap replicate随交付保存。\n\n"
        f"独立验证{v['status']}：不导入审计helper的NumPy/SciPy重算{v['native_rankdata_image_computations']}个图像-AUROC组合；"
        "全部45组原生confusion和支持度分布重算；9个固定真实位置显式枚举邻居。"
        f"支持度最大误差{v['max_support_error']:.3g}、context误差{v['max_context_error']:.3g}、GT composition误差{v['max_gt_composition_error']:.3g}。"
        f"另用与分析器不同的索引求和方式复算32个bootstrap/159列，最大差{v['max_bootstrap_error']:.3g}；"
        "全部10,000次CI分位点一致。23项单元测试及真实图像smoke通过。")
    section(27,"Frozen Gate A/B/C/D",
        table([dict(gate=k,result=("PASS" if z else "FAIL") if isinstance(z,bool) else z) for k,z in s["gates"].items()],[("Gate","gate"),("Result","result")])+
        "\nA：两种family bias mean和CI下界都>0。B：imageAUC>=.70、BA>=.62、双方recall>=.55、"
        "全体mean Delta绝对值缩小超过一半，全部满足。\n\n"
        "C：class2点估计0.451437>=.45，按原门槛PASS；class3只有418个Shallow-Win，固定为UNDERPOWERED。"
        "按用户确认的汇总规则：无powered FAIL且存在underpowered => C UNDERPOWERED。"
        "没有以置信区间或另一个AUROC定义事后替换原门槛。D：ctx_sym满足accuracy/CI/rescue/harm条件。")
    section(28,"Strong-signal flags",
        f"STRONG_SYMMETRY_SIGNAL = {str(s['strong_symmetry_signal']).upper()}\n\n"
        f"STRONG_THIRD_EVIDENCE_SIGNAL = {str(s['strong_third_evidence_signal']).upper()}\n\n"
        f"THIRD_EVIDENCE_SUPPORTED = {str(s['third_evidence_supported']).upper()}\n\n"
        "强信号是预注册辅助标志，不覆盖class3欠充分证据、不解锁训练。")
    section(29,"Scientific interpretation and limits",
        "1. **同源偏差得到支持。** shallow-source和deep-source的绝对支持方向相反，两个family bias均显著为正；"
        "对称化将全局零点偏移缩小约74.56%，并改善固定零阈值的双向召回。不能把这一比例解读为训练性能差距的解释率。\n"
        "2. **对称化有机制价值，但类别反转尚未完全解决。** 总体image AUROC、BA和anchor诊断均改善；"
        "class2的CI仍低于0.5，class3的image AUROC仍偏低且Shallow-Win不足。pooled class2/3 AUROC已超过0.5，"
        "但不能据此替换冻结的image-balanced primary；跨图混合和图内排序是不同问题。\n"
        "3. **context可提供候选之外的语义救援。** ctx_sym在Both-Wrong纠正33.61%，one-correct第三类侵入4.53%，"
        "满足本阶段第三证据门槛。这不等于统计独立、更不等于context-only就是最终第三分支。\n"
        "4. **本轮不支持直接进入模型训练。** 所有提升来自固定概率缓存上的机制探针；"
        "未证明官方final-CAM或训练后mIoU会提高。未挑选更好的context source、prediction pair、阈值或类别规则。"
        "原Phase-2B1 NOGO没有改写。\n\n"
        "按实施交付流程保留冻结合同、独立分支、可复算证据、完整报告和运行说明；PR仅供审核，停止于当前审计。")
    section(30,"Exact decision / STOP",
        "A/B通过，C UNDERPOWERED，D通过。按结果前确认的决策优先级，"
        "应使用SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED，而不是BIAS_RESOLVED。"
        "不启动Phase-2B2、不新增模型、不训练、不访问test。所有数值证据和限制已经交付，等待用户另行决定。")
    parts.append("DECISION = "+s["decision"])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="\n") as stream:
        stream.write("\n".join(parts)+"\n")
    print(str(out))


if __name__=="__main__":main()
