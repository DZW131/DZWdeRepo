"""Render the complete evidence-linked Markdown report using only the standard library."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

P='rddr_phase2b16_'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();root=Path(args.results);dest=Path(args.output)
    if dest.exists():raise FileExistsError(dest)
    def js(name):return json.loads((root/(P+name+'.json')).read_text(encoding='utf-8'))
    def rows(name):
        with (root/(P+name+'.csv')).open(encoding='utf-8') as f:return list(csv.DictReader(f))
    s=js('summary');r=js('runtime');v=js('verification');i=js('identity_audit');b=js('bf16_smoke');d=js('detach_audit');ids=js('mathematical_identities')
    assert v['status']=='PASS'
    metrics=rows('teacher_metrics');utility=rows('gradient_semantic_utility');boot=rows('bootstrap')
    lookup={(x['loss'],x['stratum']):x for x in utility}
    def num(value):
        try:
            x=float(value)
            if x!=0 and abs(x)<.001:return f'{x:.6e}'
            return f'{x:.6f}'
        except (TypeError,ValueError):return str(value)
    def percent(value):return f'{100*float(value):.4f}'
    def table(rs,cols,pct=()):
        lines=['| '+' | '.join(label for key,label in cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
        for row in rs:
            cells=[]
            for key,_ in cols:
                val=row[key]
                cells.append(percent(val) if key in pct else (str(val) if key in ('estimator','stratum','loss','teacher','parameter','path','metric','power','images','targets','repair','harm','net_repair','zero_gradient_images','finite') else num(val)))
            lines.append('| '+' | '.join(cells)+' |')
        return '\n'.join(lines)+'\n'
    def link(name,ext='csv'):return f'[{P+name+"."+ext}](../audit/results/rddr_phase2b16/{P+name+"."+ext})'
    md=[]
    def sec(title,text):md.append('## '+title+'\n\n'+text.rstrip()+'\n')
    md.append('# RDDR Phase-2B1.6 Trainability & Integration Audit\n\nSymmetric Adjudication · BCSS validation · zero-update · C0 Full25 seed42\n')
    md.append('结论：**teacher 信号保留，但本轮预注册的 conflict-weighted KL 消费方式未通过梯度安全门。**\n\n'
              'Gate A/B/C/D = **PASS / PASS / FAIL / PASS**。没有启动 Full25，没有选择 lambda，没有更新或保存任何模型权重，没有访问 test/LUAD。\n')
    sec('1. Provenance、冻结资产与样本范围',
        f'- Pure A0：`{r["a0_commit"]}`。分支：`feature/rddr-phase2b16-trainability`。\n'
        f'- 实际 GPU 审计代码：`{r["code_commit"]}`；独立复算代码：`{v["source_commit"]}`。后续提交仅补充报告/交付。\n'
        f'- checkpoint：`{r["checkpoint"]}`\n- checkpoint SHA256：`{r["checkpoint_sha256"]}`\n'
        f'- Phase2B1 cache：`{r["native"]}`\n- SHA256：`{r["native_sha256"]}`\n'
        f'- Phase2B1.5 cache：`{r["derived"]}`\n- SHA256：`{r["derived_sha256"]}`\n'
        f'- 本轮梯度观测 SHA256：`{r["observations_sha256"]}`。服务器 `{s["run"]}` 保留完整 NPZ 和逐图梯度统计。\n\n'
        '全部 **3418** validation 图像均执行 forward/backward。native28 共 2,679,712 个像素全部进入 loss 分母；'
        '2,479,143 个有效前景像素进入 GT 诊断，分布于 3416 张图像。另 2 张图像并未被丢弃，只是不含 native28 前景 GT。\n\n'
        'GT 0–3；background/ignore 只排除于 GT 指标，不排除于 loss。本文所有语义指标均为 native28 四前景类 pooled confusion matrix，'
        'absent-union class=NA、不进入 macro mean。**不能与正式224分辨率、TTA/fusion后的 A0 mIoU 直接相减。**\n\n'
        '原始 network、train_sshr.py、tool/infer_fun.py、iouutils 和 optimizer 文件零修改。合同见 [冻结合同](rddr_phase2b16_contract.md)。')
    sec('2. Exact commands 与环境',
        f'运行目录：`/home/duyanhong/DZWdeRepo-rddr-phase2b16`。GPU `{r["gpu"]}`；PyTorch `{r["torch"]}`；NumPy `{r["numpy"]}`。\n\n'
        '```bash\n'+r['command']+'\n'+s['analysis_command']+'\n'+v['command']+'\n```\n\n'
        'BF16 forward、FP32 softmax/KL；batch1；cudnn.benchmark=False，matmul precision=none，conv precision=tf32，与冻结 cache backend 一致。'
        '每个 loss 的系数均为1。本轮没有 optimizer，也没有正式训练命令。')
    sec('3. Teacher/q 重放一致性',
        table([dict(parameter=k,max_abs=x) for k,x in r['parity_max_abs'].items()],[('parameter','量'),('max_abs','最大绝对误差')])+
        '\nT_SS/T_SD/T_DS/T_DD、Delta_sym、wD、teacher 均逐位一致；q 的 5.96e-8 差别来自冻结 NumPy division 与 Torch division 的 FP32 舍入，'
        '小于合同1e-7。主审计直接使用原始缓存 q，没有重定义。重新前向 p_s、p_d 对缓存最大误差均为0。\n\n'
        f'独立 FP64 公式复算 q 与冻结 FP32 的差为 {v["errors"]["q_float64_vs_frozen_max_abs"]:.6e}；这是额外跨精度检查，'
        '不用于替代或放宽前面的原实现1e-7门限。')
    sec('4. 三个冻结 loss',
        '```text\np_rect = softmax(L_rect.float())\nKL_i = sum_k t_ik * (log(t_ik+1e-8) - log(p_rect_ik+1e-8))\n'
        'U   = mean_i KL(sym_teacher || rect)\nFA  = sum_i q_i*KL(fixedavg_teacher || rect)/(sum_i q_i+1e-8)\n'
        'CCA = sum_i q_i*KL(sym_teacher || rect)/(sum_i q_i+1e-8)\n```\n\n'+
        table([dict(loss=k,**val) for k,val in s['loss_summary'].items()],[('loss','Loss'),('mean','逐图均值'),('min','最小'),('max','最大')])+
        '\n所有 native28 位置进入 loss；GT 不进入公式。主结果按单图分母归一化，batch20 smoke 按整个 batch 分母归一化，二者原始梯度幅度不能不加说明地互比。')
    sec('5. q 的角色与数学限制',
        '`q = JS(p_s,p_d)/ln2` 保持原公式、原缓存、原 quintile 和 Top20 定义。它只分配 loss 权重，不改变 feature/context 幅度。\n\n'
        '对同一 teacher、q>0 时：`g_CCA(i) = [784*q_i/(sum_j q_j+eps)] * g_U(i)`。这是正标量缩放，'
        '**可以重新分配梯度大小，但不能修正该像素的方向符号**。\n\n'
        f'有效前景 q>0 数量={ids["q_positive_targets"]:,}，q=0数量={ids["q_zero_targets"]}。正比例梯度最大 FP32 误差='
        f'{ids["cca_u_gradient_positive_scaling_max_abs"]:.6e}；dM 符号不一致像素={ids["cca_u_dm_sign_mismatch"]}。'
        'U/CCA BenefitRate 相同不是两个独立发现，也不是实现错误。')
    sec('6. Detach 和梯度范围',
        'teacher/q 全部 detached；真实 batch20 从本批 p_s/p_d 重建 teacher 后，teacher-source 的 p_s.grad、p_d.grad 均为 None。'
        'raw CAM 复用 ic1 时，ic1 的 student 梯度合法存在，不能误当作 teacher 泄漏。\n\n'
        '只允许 HFRM28_1 的 context_conv、两层 veto_mlp、gamma_context/gamma_veto，以及 ic1.weight/bias 七个 tensor 求导。'
        '其余参数梯度全部 None，全部模块 eval，BN buffers 不变。程序同时阻止 optimizer 构造和 checkpoint 写出。\n\n'+link('detach_audit','json'))
    sec('7. Teacher / FixedAvg / Rect 的语义指标',
        table(metrics,[('estimator','分布'),('accuracy','Accuracy %'),('miou','mIoU %'),('dice','Dice %'),('nll','NLL'),('brier','Brier')],('accuracy','miou','dice'))+
        '\n'+table(metrics,[('estimator','分布')]+[(f'class{k}_iou',f'Class{k} IoU %') for k in range(4)],tuple(f'class{k}_iou' for k in range(4)))+
        '\nSym teacher 比 FixedAvg **+1.9651 pp mIoU**，但比当前 rectified distribution **-4.4724 pp mIoU**；'
        'accuracy 比 rect 低3.2405 pp。Gate A 检验的是“优于 FixedAvg”，不是“优于当前 student”。'
        'teacher NLL 较低而 Brier 较高，说明分类正确性与概率校准/置信度不能混为一个指标。\n\n'+link('teacher_metrics'))
    sec('8. Repair / Harm / NetRepair',
        table(rows('teacher_transition'),[('stratum','分组'),('teacher','Teacher'),('repair','Repair'),('harm','Harm'),('net_repair','Net count'),('net_repair_rate','Net rate pp')],('net_repair_rate',))+
        '\n整体 teacher 修复88,290个、损伤168,626个，净 -80,336（-3.2405 pp）；比 FixedAvg 的 -108,342 好，但仍是净负向。\n\n'
        f'`NetRepair rate = teacher accuracy - rect accuracy` 恒等式最大误差={ids["net_repair_accuracy_identity_max_abs"]:.6e}。'
        '相对共同 rect 参照，teacher-vs-fixed 的 NetRepair 差就是 accuracy 差，不能将二者重复计算成独立证据。')
    sec('9. GT probability advantage 与 q 分层 teacher utility',
        table([x for x in rows('teacher_advantage') if x['stratum'] in ('all','Top20','Bottom80','Rect_Correct','Rect_Wrong')],
              [('stratum','分组'),('teacher','比较'),('mean','Mean ΔP_GT'),('median','Median'),('positive_fraction','正向 %'),('negative_fraction','负向 %')],('positive_fraction','negative_fraction'))+
        '\n'+table(rows('q_strata'),[('stratum','分位'),('teacher','Teacher'),('net_repair_rate','NetRepair pp'),('teacher_advantage','ΔP_GT'),('mean_kl','Mean KL')],('net_repair_rate',))+
        '\n高冲突只表示分歧/监督需求，不自动保证 teacher 在该位置可靠。本轮不根据这些统计修改 q。完整 advantage 分层见 '+link('teacher_advantage')+'。')
    sec('10. Logit gradient',
        table([x for x in rows('logit_gradient') if x['stratum'] in ('all','Top20','Bottom80')],
              [('loss','Loss'),('stratum','分组'),('mean_G','Mean pixel L2'),('rms','RMS'),('max_abs','Max abs'),('finite','Finite')])+
        '\n直接对 FP32 `L_rect` 求导；返回形状为1×4×28×28。独立解析验证使用带 epsilon 的准确导数，'
        '而不是将其近似为 p-t。`a=t*p/(p+eps); g=w*(p*sum(a)-a)`。\n\n'
        f'FP64 解析梯度与实测 FP32 autograd 最大误差={v["errors"]["analytic_gradient_max_abs"]:.6e}。')
    sec('11. 一阶 GT-margin 与 tie 处理',
        '`M=L_GT-max_nonGT L`；`DeltaL=-g`；`dM=DeltaL_GT-max(DeltaL_k | k在当前max非GT并列集合)`。'
        '这里取的是方向导数，不随意选择一个 tie index。\n\n'
        f'前景共 {ids["tied_competitor_pixels"]:,} 个像素出现并列最大非GT logit（Top20={ids["tie_counts"]["Top20"]}）。'
        '使用严格 >0/<0，不调整数值阈值。\n\n'
        '**dM<0 表示无穷小辅助梯度降低 GT margin，不等于该像素已经预测错误，更不等于一次真实训练后必然改变类别。**'
        '本轮禁止任何真实参数更新，诊断属于已预注册的局部安全代理指标。')
    sec('12. BenefitRate / HarmRate 与 Correct-Wrong 安全性',
        table([x for x in utility if x['stratum'] in ('all','Top20','Bottom80','Rect_Correct','Rect_Wrong')],
              [('loss','Loss'),('stratum','分组'),('benefit_rate','Benefit %'),('harm_rate','Harm %'),('mean_dm','Mean dM'),('median_dm','Median dM')],('benefit_rate','harm_rate'))+
        '\nCCA 整体 Benefit **17.9611%**、Harm **82.0389%**；Top20 Benefit **31.2450%**、Harm **68.7550%**。'
        'Rect_Wrong 有86.8583%得到正向 margin 推力，但 Rect_Correct 有97.3899%的 margin 下降，后者占总体多数。'
        '这可能包含向软 teacher 置信度收缩的效应；不能直接等价为97.39%的正确像素被分错。\n\n'+link('gradient_semantic_utility'))
    sec('13. Conflict localization',
        table(rows('gradient_localization'),[('loss','Loss'),('Top20_mean_G','Top20 mean G'),('Bottom80_mean_G','Bottom80 mean G'),('Top20_over_Bottom80','Top/Bottom'),('Q1_mean_G','Q1'),('Q5_mean_G','Q5')])+
        '\nCCA Top/Bottom=5.3713，高于 U 的2.3926；Q5>Q1，所以 Gate B PASS。'
        '但是权重更集中并未改变逐像素梯度方向；本次加权后整体和Top20的平均 dM 比 U 更负。')
    sec('14. 四类梯度安全与样本充分性',
        table([x for x in rows('per_class_gradient') if x['loss']=='CCA'],
              [('stratum','类别'),('targets','像素'),('images','图像'),('benefit_rate','Benefit %'),('harm_rate','Harm %'),('mean_dm','Mean dM'),('gradient_norm','Mean G'),('power','Power')],('benefit_rate','harm_rate'))+
        '\n充分性阈值为≥500前景像素且≥30图像。class3 为145,803像素/384图像，**不再沿用上一轮 Shallow-Win 子集的 underpowered 标记**。'
        '四类均充分、四类 mean dM 均负；加上 all/Top20，正向为 **0/6**，低于要求的≥5/6。')
    sec('15. F28_rect feature gradient',
        table(rows('feature_gradient'),[('stratum','分组'),('rms','RMS'),('mean_pixel_L2','Mean pixel L2'),('max_abs','Max abs'),('finite','Finite')])+
        '\n所有3418张图像 feature gradient 有限且非零；RMS按512通道统计，pixel L2对512通道求范数。'
        'GT分层只用于统计，不参与反向图。')
    sec('16. Parameter gradient',
        table(rows('parameter_gradient'),[('parameter','参数'),('pooled_rms','Pooled RMS'),('max_abs','Max abs'),('mean_nonzero_fraction','Mean nonzero fraction'),('zero_gradient_images','全零梯度图像数'),('finite','Finite')])+
        '\nPooled RMS = sqrt(sum_image sum_param grad² / (images×parameter_count))。只记录，不进行 optimizer step。'
        '逐图、逐参数完整记录见 '+link('parameter_per_image')+'。')
    sec('17. Gradient path attribution',
        table(rows('gradient_path_attribution'),[('path','分支'),('pooled_L2','Pooled L2'),('L2_fraction','L2/total'),('squared_energy_share','平方能量占比 %'),('zero_gradient_images','全零图像数')],('squared_energy_share',))+
        '\ncontext/semantic/head 三路均连通，并非只有 head 收到梯度。注意 semantic 平方能量占比仅约 **0.05945%**，'
        '明显小于 context 38.2517% 和 head 61.6889%；这不触发本轮仅要求非零的 Gate D，'
        '但也不能宣称各路得到了同等强度学习。不同参数量/参数化影响此比例，它不是因果贡献率。L2比例本身不要求和为1；平方能量比例和为1。')
    sec('18. 数值稳定性与固定样本',
        '预先冻结32个等距图像索引；seed42从剩余索引抽128个、不放回。160张均包含于全量3418反向审计，并额外重放全部原始 forward 返回tensor，逐位一致。'
        '全部loss/logits/logit gradients/feature gradients/parameter gradients有限，未发生NaN/Inf。\n\n'
        +link('selection','json')+'；'+link('runtime','json')+'。')
    sec('19. Batch20 BF16 smoke 与资源',
        f'- 固定 deterministic32 的前20张，224×224，BF16 forward / FP32 loss，真实 teacher construction + backward。\n'
        f'- Loss={b["loss"]:.6f}；耗时={b["seconds"]:.6f} s。\n'
        f'- Peak allocated={b["peak_cuda_allocated_bytes"]/1024**3:.4f} GiB；reserved={b["peak_cuda_reserved_bytes"]/1024**3:.4f} GiB；预算22 GiB，通过。\n'
        f'- 全量 teacher重放={r["parity_seconds"]:.3f}s；全量3418梯度审计={r["gradient_seconds"]:.3f}s；本轮runner={r["total_seconds"]:.3f}s。\n\n'
        '资源值为服务器程序计时器与 CUDA memory API 的实测，不是25epoch训练时间预测。**本轮只开放HFRM28_1/ic1反向，'
        'backbone没有梯度图；不能拿1.29GiB推断完全解冻训练的显存。**\n\n'+link('bf16_smoke','json'))
    sec('20. Inference identity / zero-update identity',
        f'- 原始 state_dict 所有parameter+buffer前后哈希：`{i["state_before"]}`（相同）。\n'
        f'- checkpoint文件前后SHA：`{i["checkpoint_sha_after"]}`（相同）。\n'
        f'- 固定160张完整官方推理、共{i["official_before"]["pixels"]:,}像素，原始预测前后SHA：`{i["official_before"]["prediction_sha256"]}`（相同）。\n'
        '- 原始三路TTA、BCSS class thresholds=[0.8,0.9,0.8,0.6]、CAM融合=[0.6,0.2,0.2]、插值、normalization、presence与argmax流程未变。\n'
        '- 只在audit中限制Dataset到固定160，并拦截最终scores入口计算原始prediction哈希，避免background overwrite影响identity证据。\n'
        '- 原始checkpoint strict load：missing_keys=[]，unexpected_keys=[]。\n\n'
        'CCA helper 不接入model.forward或infer，off不构造teacher/support；on只增加审计loss和梯度，零step后输出不变。'
        '没有为未来训练新增CLI或更改训练入口。本轮证实当前零更新路径的identity，不是尚未实现的未来训练runner测试。\n\n'+link('identity_audit','json'))
    sec('21. Third-evidence holdout 与为什么不用 forward 注入',
        'teacher严格只有 `wS*p_s+wD*p_d`；没有ctx_sym、第三类恢复、Both-Wrong detector、额外loss。'
        'Both-Wrong只作为GT事后分析分组，未被训练模块读取。\n\n'
        '历史 Phase1 direct feature manipulation、Phase2A context amplitude manipulation 和 Phase2B1 naive anchor fusion 的风险背景下，'
        '本轮只检查teacher-only supervision；不做feature replacement/context residual replacement/direct anchor overwrite/fixed inference fusion。'
        '本轮失败也不构成回退到上述路径的授权。')
    sec('22. 10,000次配对 image bootstrap 与独立验证',
        table(boot,[('metric','估计量'),('estimate','Point'),('ci_low','95% CI lower'),('ci_high','95% CI upper')])+
        '\n概率差/accuracy/mIoU表内单位是0–1；换算pp需×100；dM维持原始logit导数单位。image-level、paired、seed42，'
        '每次重采样重新pool confusion/count/sum，不是对像素独立bootstrap，也不是平均逐图mIoU。\n\n'
        f'独立NumPy复算 `{v["status"]}`：{len(v["checks"])}项通过。完整复算不导入原loss/analyzer，'
        '使用FP64解析梯度、显式真值/预测mask构建confusion、非GT索引gather处理ties，并以gather-sum重做全部bootstrap。\n\n'+
        table([dict(parameter=k,max_abs=x) for k,x in v['errors'].items()],[('parameter','独立校验'),('max_abs','最大绝对差')])+
        '\n关键六组CCA mean_dM的95% CI均完全小于0；不是class3样本不足或bootstrap跨零。完整证据见 '+link('verification','json')+' 与 '+link('bootstrap_replicates')+'。')
    sec('23. Gate A/B/C/D 判定',
        '| Gate | 结果 | 依据 |\n| --- | --- | --- |\n'
        '| A Teacher superiority | PASS | 优于FixedAvg；ΔmIoU +1.9651pp，95%CI [+1.8105,+2.1300]pp；accuracy/相对NetRepair也更好。后两者不是独立证据。 |\n'
        '| B Conflict localization | PASS | Top20>Bottom80，5.3713>2.3926，Q5>Q1。 |\n'
        '| C Gradient semantic utility | FAIL | all/Top20 Benefit<Harm，关键mean_dM正向0/6；四类统计充分。 |\n'
        '| D Engineering | PASS | 三路非零且有限，detach正确，batch20通过、显存符合预算，零step、权重/推理不变。 |\n\n'
        '使用已批准的优先级D>A>B>C。这里不是工程失败，也不是teacher相对FixedAvg失去优势；是局部KL梯度安全性失败。')
    sec('24. Secondary preference flag',
        '**ADJUDICATION_TEACHER_PREFERRED = TRUE**。CCA相对FA的整体BenefitRate +0.1134pp，Top20 mean_dM +1.7321e-4。'
        '这是“比FA更好”，不是“绝对安全”：两者整体/Top20 mean_dM都为负。不能用secondary flag覆盖Gate C。')
    sec('25. Scientific interpretation、限制和交付',
        '本轮支持 **Outcome B**：symmetric adjudication的相对语义信号成立，但当前KL-style消费方式没有通过已冻结的GT-margin安全标准。\n\n'
        '1. current rect在native28分类上比teacher强；向teacher贴近会同时纠正一些错误、压低大量正确像素的GT margin。\n'
        '2. q强调冲突幅度，不是teacher正确性；它不能改变U/CCA的像素梯度符号。本次局部加权更集中，但总体方向仍不安全。\n'
        '3. symmetric teacher较FA好，不能自动推导其优于student，更不能推导Full25增益。\n'
        '4. “margin下降”可能包含正常软目标置信度收缩。此次结论是预注册代理门失败，**不是已经测得Full25 mIoU下降**，也不证明所有lambda或所有KL训练都失败。\n'
        '5. 当前只测单位系数辅助loss的局部梯度，没有测它与SSHR分类loss的合成方向、真实更新或长期泛化。不能绕过本合同去做lambda搜索。\n\n'
        '本轮完成：独立A0分支、GT-blind helper、单元/真实GPU安全测试、全量3418诊断、10k bootstrap、独立复算、CSV/JSON/Markdown。'
        '模型和训练/推理/metric源码未变；没有数据删除或覆盖。仅新增约152MiB服务器梯度观测与文本统计。\n\n'
        '如继续研究，需用户另行定义teacher consumption假设与预注册协议；本报告不自行设计、测试或推荐某个未验证补丁。'
        '不选择lambda，不启动Full25，不测试其他seed/LUAD/test。命令/目录见 [执行README](README_rddr_phase2b16.md)。')
    sec('26. Exact decision 与停止条件',
        '已完成审计并停止。保留symmetric adjudication信号证据；**本轮conflict-weighted KL路线不进入Full25**。'
        '下一阶段必须重新明确并审核执行方案。\n\n'+link('summary','json'))
    md.append('DECISION = '+s['decision'])
    text='\n'.join(md)+'\n';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',encoding='utf-8',newline='\n') as f:f.write(text)
    print(json.dumps(dict(report=str(dest),sha256=hashlib.sha256(text.encode()).hexdigest(),decision=s['decision']),ensure_ascii=False))


if __name__=='__main__':main()
