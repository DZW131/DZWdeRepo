"""Deterministic, stdlib-only delivery renderer; never reruns a scientific audit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

P = 'rddr_phase2b110_'


def number(x, digits=4):
    v = float(x)
    return f'{v:.{digits}f}' if math.isfinite(v) else 'NA'


def count(x):
    return f'{int(float(x)):,}'


def percent(x):
    return number(float(x) * 100) + '%'


def table(headers, rows):
    def line(row):
        return '| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |\n'
    return line(headers) + line(['---'] * len(headers)) + ''.join(line(r) for r in rows) + '\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(run):
    def js(name):
        return json.loads((run / (P + name + '.json')).read_text(encoding='utf-8'))

    def rows(name):
        with (run / (P + name + '.csv')).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    s, rt, ver, identity = (js(k) for k in ('summary', 'runtime', 'verification', 'identity_audit'))
    tests = (run / (P + 'tests.txt')).read_text(encoding='utf-8')
    assert ver['status'] == 'PASS' and all(ver['checks'].values())
    assert 'Ran 44 tests' in tests and tests.rstrip().endswith('OK') and 'skipped=' not in tests
    assert s['decision'] == ver['decision'] == 'RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED'
    assert [s['gate_' + k] for k in 'ABCD'] == ['PASS', 'FAIL', 'FAIL', 'FAIL']
    assert s['RESIDUAL_THIRD_EVIDENCE_SIGNAL'] and not s['STRONG_RESIDUAL_DEEP_RECOVERY_SIGNAL']
    assert not any(rt[k] for k in ('model_instantiated', 'network_forward', 'backward',
                                  'optimizer_created', 'optimizer_steps', 'checkpoint_written',
                                  'new_recovery_gate', 'threshold_search', 'test_access',
                                  'luad_access', 'training_split_access'))
    ci_rows = rows('bootstrap')
    cis = {r['metric']: r for r in ci_rows}
    assert all(int(r['resamples']) == int(r['valid_resamples']) == 10000 for r in ci_rows)

    def ci(key, as_percent=False):
        r = cis[key]
        fmt = percent if as_percent else number
        return '[' + fmt(r['ci_low']) + ', ' + fmt(r['ci_high']) + ']'

    def section(n, title, body):
        return f'## {n}. {title}\n\n{body.rstrip()}\n\n'

    def source(name):
        return f'[原始 CSV](../audit/results/rddr_phase2b110/{P}{name}.csv)\n\n'

    h, utility, winner = s['headroom'], s['primary_utility'], s['primary_winner']
    bw, dw, sw = s['third_evidence']
    out = ['# RDDR Phase-2B1.10 — Residual Correction Coverage & Recoverability Audit\n\n',
           '完整实验报告｜BCSS validation-only｜C0 seed42 / final Epoch25｜zero training\n\n',
           '结论摘要：潜在有益残余数量充分，但预注册主分数 `S_D_sym` 未通过恢复能力门槛；'
           '仅第三证据诊断得到支持。**这不是训练 GO，也不是已经验证的安全选择器。**\n\n']
    body = ('本轮覆盖全部 **3,418 张 validation 图像**，使用既有 native 28×28 冻结观测；'
            '不读取训练/test/LUAD split，不加载模型，不执行新的网络 forward/backward。'
            '所有位置计数均是 native28 特征位置，不是 224×224 分割像素计数。\n\n'
            f'- 纯 A0 基线：`{rt["a0"]}`。\n'
            '- 分支：`feature/rddr-phase2b110-residual-coverage`；PR 目标 `baseline/official-a0`。\n'
            f'- 主审计运行 commit：`{rt["code_commit"]}`。\n'
            f'- 独立验证 commit：`{ver["code_commit"]}`；后续报告提交不改变上述运行来源。\n'
            f'- 冻结合同 SHA256：`{rt["contract_sha256"]}`。\n\n'
            '输入路径与运行前后相同的 SHA256：\n\n')
    for key, path in rt['paths'].items():
        body += f'- `{key}`：`{path}`\n  SHA256：`{rt["source_sha256"][key]}`\n'
    body += ('\n准确执行命令（历史记录，既有 output 不可覆盖；复跑须使用新目录）：\n\n'
             '```bash\ncd /home/duyanhong/DZWdeRepo-rddr-phase2b110\n' + rt['command'] + '\n'
             + ver['command'] + '\n'
             'RDDR_PHASE2B110_RUN=/home/duyanhong/experiments/RDDR_PHASE2B110/formal_r1 '
             '/home/duyanhong/miniconda3/envs/sshr5090/bin/python -m unittest discover -s tests '
             "-p 'test_rddr_phase2b110*.py' -v\n```\n\n")
    replay = rt['replay']
    body += table(['实测项目', '结果'], [
        ['环境', f'{replay["gpu"]}; PyTorch {replay["torch"]}; NumPy {rt["numpy"]}'],
        ['主审计运行时间', number(rt['total_seconds']) + ' s'],
        ['概率张量 GPU 重放', number(replay['seconds']) + ' s（不是网络推理）'],
        ['概率重放峰值 allocated / reserved', f'{number(replay["allocated_bytes"] / 2**20)} / {number(replay["reserved_bytes"] / 2**20)} MiB'],
        ['主流程 FP32 support / context 最大误差', '0 / 0（四个 support、三个 context 全量重放）'],
        ['q 重算最大误差', f'{replay["errors"]["q"]:.9e} ≤ 1e-7；保留原缓存 q'],
        ['独立 FP64 context 对 FP32 缓存', f'{ver["errors"]["independent_FP64_context"]:.9e} < 1e-6；不同归约顺序交叉检查'],
        ['独立排名 / context 指标最大误差', f'{ver["errors"]["rank_metrics"]:.3e} / {ver["errors"]["context_metrics"]:.3e}'],
        ['工程验证', f'44/44 unit+integration tests；{len(ver["checks"])}/{len(ver["checks"])} 独立检查；所有输出有限'],
        ['模型 / 训练操作', 'model=0, network forward=0, backward=0, optimizer step=0, checkpoint write=0'],
    ])
    body += ('**Identity 证据边界：**本轮新测的是全部输入文件 SHA 前后一致、A0 原始源码不变，'
             '以及冻结张量/计数重放。既有 Phase2B1.9 的 state/BN/160-image prediction identity 原样继承，'
             '**没有**把它们标作本轮新运行的模型等价测试。checkpoint 只计算 SHA，不实例化/加载网络。\n\n'
             '[identity 记录](../audit/results/rddr_phase2b110/rddr_phase2b110_identity_audit.json)；'
             '[独立验证](../audit/results/rddr_phase2b110/rddr_phase2b110_verification.json)；'
             '[44 项测试日志](../audit/results/rddr_phase2b110/rddr_phase2b110_tests.txt)。')
    assert identity['new_checks']['new_state_bn_prediction_test'] is False
    out.append(section(1, 'Provenance / SHA / commands', body))

    out.append(section(2, 'Phase-2B1.9 frozen status',
        table(['Gate', *list('ABCDEFG')], [['冻结结果', *[s['prior_phase2b19_gates'][k] for k in 'ABCDEFG']]]) +
        f'冻结决定：`{s["prior_phase2b19_decision"]}`。本轮不改判旧实验。\n\n'
        '旧 Gate E 的 Raw-Wrong beneficial coverage 未达 40%。此前较低 HHCR、较高 ShallowProtection '
        '不能概括为“已经完整安全”：负向 Shallow-Win dM 幅度增加、class3 安全证据不足的限制仍保留。'
        '本轮只诊断 residual coverage，不能用新的分组统计倒推旧的安全问题已解决。'))

    out.append(section(3, 'Exact Gate-E deficit',
        f'`N_RW = {h["RawWrong"]}`；`B_ADT = {h["ADT_beneficial"]}`。\n\n'
        f'全分母 benefit rate = `{h["ADT_benefit_rate"]:.16f}` = {percent(h["ADT_benefit_rate"])}。'
        f'距离 40% 的连续比例差为 **{number(100 * (.4 - h["ADT_benefit_rate"]), 8)} pp**。'
        '以下计数使用精确整数，不使用打印后截断的 0.3558646371。'))
    out.append(section(4, 'Required additional beneficial count',
        '```text\nTarget = ceil(2 * N_RW / 5)\n'
        f'       = {h["target_beneficial"]}\n'
        f'RequiredAdditionalBenefit = {h["target_beneficial"]} - {h["ADT_beneficial"]} = {h["required_additional"]}\n```\n\n'
        f'因此至少需要 **{count(h["required_additional"])}** 个额外有益位置；折算全 Raw-Wrong 分母为 '
        f'{number(100 * h["required_additional_rate"], 8)} pp。'
        '该整数折算率与上一节连续差的微小区别来自 ceil，不是计算冲突。'))

    out.append(section(5, 'Residual population',
        '`m_D = 1[Delta_sym > 0]`；`R = {m_D=0}`；`R_RW = R ∩ {raw wrong}`，'
        '诊断人口限定 GT 0–3，背景4/ignore255不参与。既有 m_D 只重放，不构造恢复 gate。\n\n' +
        table(['Population', '位置数', '有该人口的图像数'],
              [[r['population'], count(r['count']), r['images']] for r in rows('residual_counts')]) +
        '3,418 张全部读取；其中 3,416 张在 native28 有前景。人口图像数不同不是静默丢图。'))
    out.append(section(6, 'Rejected Deep-Win / Both-Wrong counts',
        f'`R_RW = {count(h["rejected_Deep_Win"])} Rejected Deep-Win + '
        f'{count(bw["targets"])} Rejected Both-Wrong = 435,185`，与冻结记录 exact replay。\n\n'
        f'Both-Wrong 占 residual Raw-Wrong 的 **{percent(bw["targets"] / 435185)}**。'
        f'另有 {count(sw["targets"])} 个 Rejected Shallow-Win，用于 one-correct 对照；'
        '它们 raw 正确，不能加进 R_RW 分母。'))
    out.append(section(7, 'Residual headroom',
        table(['量', '结果'], [
            ['Residual beneficial / harmful / zero', f'{count(h["residual_beneficial"])} / {count(h["residual_harmful"])} / {h["residual_zero"]}'],
            ['CoverageHeadroom = beneficial / ALL Raw-Wrong', percent(h['coverage_headroom'])],
            ['95% CI of CoverageHeadroom', ci('CoverageHeadroom_rate', True)],
            ['原始 N_RW 下 count-equivalent 95% CI', ci('ResidualBeneficial_count_equivalent')],
            ['HeadroomOverGap', count(h['headroom_over_gap'])],
            ['只从 rejected Deep-Win 补足缺口所需比例', percent(h['required_fraction_of_rejected_Deep_Win'])],
            ['Residual beneficial prevalence（含 zero 分母）', percent(cis['ResidualBeneficial_prevalence']['estimate'])],
            ['该 prevalence 95% CI', ci('ResidualBeneficial_prevalence', True)],
        ]) + 'Gate A 的分母始终是所有 Raw-Wrong，不是 residual-only。理想选择器的局部导数有益机会数量充足，'
        '但这只是 arithmetic/local-derivative headroom，**不等于**这些位置经有限步训练后一定转对，更不等于 mIoU 增益。'))
    out.append(section(8, 'Primary S_D hypothesis',
        '`S_D_sym = 0.5(T_DS + T_DD)`；`S_S_sym = 0.5(T_SS + T_SD)`；'
        '`Delta_sym = S_D_sym - S_S_sym`。主假设是被相对 support 拒绝的位置中，'
        '绝对 deep support 越高越能识别有益的 deep-transfer。GT 不进入上述 score。'
        '本轮始终使用正向 S_D：没有取反、阈值化、top-k、温度或事后替换。'))
    out.append(section(9, 'Residual beneficial / harmful definition',
        '在 R_RW 上，冻结 `UDT dM > 0` 为 beneficial，`< 0` 为 harmful，`==0` 单列。'
        '用冻结 FP32 raw logits/gradient，FP64 累积 `v=-g` 的 GT-vs-max-competitor margin 方向导数；'
        '遇到最大 competitor 并列时采用方向导数的 exact max-tie 规则。没有重新 backward。\n\n'
        f'正类 {count(utility["positive"])}；负类 {count(utility["negative"])}；4 个零值仅在二分类排名中排除，'
        f'二分类 prevalence = {percent(utility["prevalence"])}，95% CI {ci("ResidualBeneficial_binary_prevalence", True)}。'
        'GT 只参与人口/标签/指标构造；这是一种回顾性诊断，不是可直接部署的 GT-blind 选择人口。'))

    for n, title, r, population in [(10, 'S_D residual utility AUROC', utility, 'residual_utility'),
                                   (11, 'S_D rejected winner AUROC', winner, 'rejected_winner')]:
        body = table(['指标', '结果'], [
            ['Primary image-balanced AUROC', number(r['image_auroc'])],
            ['95% CI', ci(population + ':all:S_D_sym:image_AUROC')],
            ['Pooled AUROC', number(r['auroc'])], ['Pooled AUPRC (Average Precision)', number(r['auprc'])],
            ['Positive prevalence', percent(r['prevalence'])],
            ['正 / 负 / binary 总数', f'{count(r["positive"])} / {count(r["negative"])} / {count(r["targets"])}'],
            ['Dual-label eligible images / images with targets', f'{r["eligible_images"]} / {r["images_with_targets"]}'],
        ])
        body += ('Gate B FAIL：主 image AUC 约为随机水平，且下界不高于 0.50；不能以 pooled AUC '
                 '或其它 score 替代主指标。' if n == 10 else
                 '正类是 Rejected Deep-Win，负类是 Rejected Shallow-Win；仅 exactly-one-correct conflict。'
                 '下界 >0.50，但 image AUC <0.65，因此 Gate C FAIL，而不是完全没有排名信号。')
        out.append(section(n, title, body))
    body = ('所有方向按合同冻结：Delta 不取绝对值/不翻符号；q 使用原缓存；confidence = max(pd)-max(ps)；'
            'entropy = H(ps)-H(pd)，H=-sum p log(p+1e-8)，单位 nats。\n\n')
    controls = rows('score_controls')
    for population in ('residual_utility', 'rejected_winner'):
        body += f'### {population}\n\n' + table(
            ['Score', 'Pooled AUROC', 'Image AUROC', 'Image AUROC 95% CI', 'AUPRC'],
            [[r['score'], number(r['auroc']), number(r['image_auroc']),
              ci(population + ':all:' + r['score'] + ':image_AUROC'), number(r['auprc'])]
             for r in controls if r['population'] == population and r['group'] == 'all'])
    body += ('q 的 utility image AUC 约0.9067，但区分 rejected winner 只有约0.5036。'
             '因此“需要纠正”与“应该信任 deep 而不是 shallow”在本次人口中是不同问题；'
             '高 utility AUC 不能证明层级安全性。confidence/Delta 等对照也不能取代失败的 S_D 主门槛，'
             '没有据此组合新 score 或建议阈值。\n\n' + source('score_controls'))
    out.append(section(12, 'Delta / q / confidence / entropy controls', body))

    for n, kind in ((13, 'beneficial'), (14, 'harmful')):
        body = table(['来源', 'Count', '该 utility 内占比', 'Mean q', 'Mean S_D', 'Mean Delta', 'Mean deep confidence'],
                     [[r['source'], count(r['count']), percent(r['fraction_of_utility']), number(r['mean_q']),
                       number(r['mean_S_D']), number(r['mean_Delta']), number(r['mean_deep_confidence'])]
                      for r in rows(kind + '_composition')])
        body += ('有益残余以 missed Deep-Win 为主（63.6460%），但仍有36.3540%来自 Both-Wrong 的局部 margin 改善。'
                 '局部 margin 增加并不要求 deep 的最终 argmax 正确。' if kind == 'beneficial' else
                 '有害残余全部属于 Both-Wrong；不存在的 harmful Deep-Win 子组均值为 NA，不是0。'
                 '4 个 zero 也都来自 Both-Wrong，在 zero_composition.csv 保留；没有用 epsilon 重标标签。')
        out.append(section(n, kind.capitalize() + ' composition', body + '\n\n' + source(kind + '_composition')))

    for n, file, title in ((15, 'delta_quintiles', 'Delta quintiles'), (16, 'deep_support_quintiles', 'S_D quintiles')):
        body = ('在 R_RW 含4个 zero 上计算20/40/60/80分位；线性 quantile，'
                '`searchsorted(side=left)` 保留 score ties、不拆 ties。Q1→Q5 为低到高。\n\n'
                '冻结诊断切点：`' + ', '.join(number(x, 10) for x in s['quintile_edges'][file]) + '`。\n\n')
        body += table(['Q', 'N', '有益%', '有害%', 'Zero', 'DW%', 'BW%', 'Mean S_D', 'Mean q', 'Mean Delta'],
                      [[r['quintile'], count(r['count']), percent(r['beneficial_rate']), percent(r['harmful_rate']),
                        r['zero_count'], percent(r['Deep_Win_fraction']), percent(r['Both_Wrong_fraction']),
                        number(r['mean_S_D']), number(r['mean_q']), number(r['mean_Delta'])] for r in rows(file)])
        body += ('越靠近 Delta=0 并未出现单调增加的有益率；最负的 Q1 反而最高。'
                 if n == 15 else '绝对 S_D 最高的 Q5 有益率反而降至23.7405%，不支持“越高越有益”的主假设。')
        body += '这些分组仅描述结构，不生成 percentile gate、放宽 sign gate 或事后翻转分数。\n\n' + source(file)
        out.append(section(n, title, body))

    def strata_table(name):
        rr = rows(name)
        return (table(['Group', 'N', '有益 / 有害 / zero', '有益率', 'DW / BW', 'Dual-label images', 'Power'],
                      [[r['group'], count(r['count']), f'{count(r["beneficial_count"])} / {count(r["harmful_count"])} / {r["zero_count"]}',
                        percent(r['beneficial_rate']), f'{count(r["rejected_Deep_Win"])} / {count(r["rejected_Both_Wrong"])}',
                        r['eligible_images'], r['power']] for r in rr]) +
                table(['Group', 'S_D Image AUC', '95% CI', 'Pooled AUC', 'AUPRC', 'DW% / BW%'],
                      [[r['group'], number(r['image_auroc']), ci('residual_utility:' + r['group'] + ':S_D_sym:image_AUROC'),
                        number(r['auroc']), number(r['auprc']),
                        percent(r['Deep_Win_fraction']) + ' / ' + percent(r['Both_Wrong_fraction'])] for r in rr]))

    out.append(section(17, 'Per-class', strata_table('per_class') +
        'Power 固定为至少500正类、500负类、30张 dual-label 图像。本轮四类均 POWERED，'
        '但仅 class1 的 image AUC >0.55，未达到至少3类。此处 class3 POWERED 是 **utility 标签**的样本量结论；'
        '与旧 Phase2B1.9 稀少 Shallow-Win 的安全性人口不同，不能消除旧 UNDERPOWERED 限制。'
        '这里只保留冻结 class0–3 编码，不制定 class-specific rule。'))
    out.append(section(18, 'Boundary / interior',
        '沿用既有 boundary ≤7px / interior >7px mask，不重算边界、不调整宽度。\n\n' +
        strata_table('boundary_interior') + 'Interior image AUC=0.5285，未达 >0.60；boundary=0.3834。'
        'Interior 的相对较好结果不足以解锁训练。'))
    out.append(section(19, 'Top20 / Bottom80',
        '复用 Phase2B1 冻结 q Top20 mask；它不是在本轮 residual 上重新划出的20%。\n\n' +
        strata_table('top20_bottom80') + 'Top20 的有益 prevalence 较高，但其中 S_D 排名依然弱。'
        '这些子群为诊断，不能执行 Top20-only rescue 或利用表格选择 gate。'))

    out.append(section(20, 'Rejected Both-Wrong ctx_sym',
        '`ctx_S/ctx_D` 分别在15×15有效图像内邻域、exclude self 取概率均值；'
        '`ctx_sym=0.5(ctx_S+ctx_D)`，argmax 无温度、无阈值。\n\n' +
        table(['指标', '结果'], [
            ['Population', count(bw['targets']) + ' Rejected Both-Wrong'],
            ['Accuracy', percent(bw['accuracy'])], ['Accuracy 95% CI', ci('ctx_sym_rejected_BothWrong_accuracy', True)],
            ['Conditional 4-class mIoU', percent(bw['miou'])], ['Conditional 4-class mDice', percent(bw['dice'])],
            ['NLL (nats)', number(bw['nll'])], ['Brier (sum over 4 classes)', number(bw['brier'])],
            ['class0 / 1 / 2 / 3 IoU', ' / '.join(percent(bw[f'iou_class{k}']) for k in range(4))],
        ]) + '指标仅在 GT 定义的 rejected Both-Wrong 子集上计算，native28 四前景 confusion matrix，'
        '无背景覆盖修正；union=0 的类别为 NA、从 mean 排除；NLL=-log(p_GT+1e-8)，Brier 不除以4。'
        '**不能与官方全 validation/test mIoU 对比，也不是本轮重新评测模型的 segmentation 结果。**'))
    out.append(section(21, 'Third-class rescue',
        table(['事件', 'Count / rate'], [
            ['Context argmax 不同于 raw 和 deep', f'{count(bw["different_from_both"])} / {percent(bw["intrusion_rate"])}'],
            ['第三类且正确', count(bw['correct_third_class'])], ['第三类但错误', count(bw['wrong_third_class'])],
            ['Rescue rate（分母全部 Rejected Both-Wrong）', percent(bw['rescue_rate'])],
            ['Rescue rate 95% CI', ci('ThirdClassRescueRate', True)],
            ['Rescue precision（分母不同于 both）', percent(bw['rescue_precision'])],
            ['Rescue precision 95% CI', ci('ThirdClassRescuePrecision', True)],
            ['Third-class harm / 全部 BW', percent(bw['third_harm_rate'])],
        ]) + '在 Both-Wrong 中，任何正确 context 必然不同于两个错误候选，所以 accuracy 与 rescue rate '
        '是**同一个事件**，不是两条独立证据。Context 来自原模型邻域概率，所谓第三证据是证据形式不同，'
        '不代表新训练的独立模型或统计独立样本。'))
    out.append(section(22, 'Third-evidence harm control',
        table(['人口', 'N', 'Context accuracy [95% CI]', 'Third intrusion [95% CI]', '总错误率'], [
            [r['population'], count(r['targets']), percent(r['accuracy']) + ' ' + ci(r['population'] + ':ctx_accuracy', True),
             percent(r['intrusion_rate']) + ' ' + ci(r['population'] + ':third_intrusion', True), percent(1 - r['accuracy'])]
            for r in (dw, sw)]) +
        f'对应第三类误入数为 {count(dw["different_from_both"])} / {count(sw["different_from_both"])}，全部错误，'
        '因为 one-correct 人口已经有一个候选正确。此处 third intrusion=third harm，但不是全部 context 错误：'
        'context 选错原有候选同样可能造成伤害。**不能把8.80%/5.90%当作总错误率。**\n\n'
        'Both-Wrong 子群由 GT 定义，部署时无法直接获得；没有证明如何 GT-blind 地识别该群并避开上述污染。'
        '因此本轮不把 ctx_sym 写入 loss、gate 或实际预测。\n\n' + source('third_evidence_harm_control')))

    body = ('10000次 paired image-level bootstrap，seed42，每次从全部3418图像有放回抽样3418张，'
            '所有 endpoint 使用同一批抽样索引。无 pixel bootstrap。每个 ratio 对抽到的图像重新汇总分子/分母；'
            'image-balanced AUC 只对抽中的 dual-label eligible 图像做等权均值。'
            '区间为2.5/97.5百分位；本次所有 endpoint 均有10000个有限 replicate。\n\n'
            'Gate A 每次计算 `sum(residual beneficial)/sum(ALL Raw-Wrong) * fixed original N_RW`；'
            '将 count-equivalent 下界与**固定**原始 gap 比较，不使用 residual prevalence 分母，'
            '也不把 bootstrap 样本量变化误称为新增机会。\n\n'
            f'抽样索引 SHA256：`{s["bootstrap_rng_sha256"]}`。独立 direct-gather 与主流程权重矩阵聚合的 '
            f'replicate 最大误差 `{ver["errors"]["bootstrap_replicates"]:.3e}`，区间误差 '
            f'`{ver["errors"]["bootstrap_intervals"]:.3e}`。\n\n')
    body += table(['Endpoint（rate用0–1，count-equivalent用位置数）', 'Estimate', '95% CI'],
                  [[r['metric'], number(r['estimate'], 6), '[' + number(r['ci_low'], 6) + ', ' + number(r['ci_high'], 6) + ']'] for r in ci_rows])
    body += ('全量原始 replicates 与 summary CSV 已保存。区间反映固定 seed42/checkpoint 下的图像抽样不确定性，'
             '不是训练 seed 方差，也不覆盖后续反复选择方案产生的适应性偏差；对照区间是诊断、未做多重比较校正。')
    out.append(section(23, 'Bootstrap', body))
    out.append(section(24, 'Gate A / B / C / D',
        table(['Gate', '冻结要求', '实测', '判定'], [
            ['A', 'beneficial count ≥ gap，且 count-equivalent CI lower ≥ gap',
             f'{count(h["residual_beneficial"])} ≥ {count(h["required_additional"])}；lower={number(h["count_equivalent_ci_low"], 2)}', s['gate_A']],
            ['B', 'utility image AUC ≥0.65 且 lower>0.50', number(utility['image_auroc']) + ' ' + ci('residual_utility:all:S_D_sym:image_AUROC'), s['gate_B']],
            ['C', 'winner image AUC ≥0.65 且 lower>0.50', number(winner['image_auroc']) + ' ' + ci('rejected_winner:all:S_D_sym:image_AUROC'), s['gate_C']],
            ['D', 'interior>0.60 且≥3个 powered class>0.55', 'interior=0.5285；1/4类通过（4/4 powered）', s['gate_D']],
        ]) + 'D 没有把 UNDERPOWERED 自动当 FAIL：本轮四类都有 power，而观测到的 interior 和类数均不达标，'
        '故确为 FAIL。主门槛不允许任何 control 替代 S_D。'))
    out.append(section(25, 'RESIDUAL_THIRD_EVIDENCE_SIGNAL',
        f'`TRUE`。Rejected Both-Wrong ctx accuracy={percent(bw["accuracy"])}≥25%，'
        f'CI lower={percent(cis["ctx_sym_rejected_BothWrong_accuracy"]["ci_low"])}>20%，'
        f'rescue rate={percent(bw["rescue_rate"])}≥20%。三项按合同满足；accuracy 与 rescue 重复事件的限制如第21节。'
        '这是 secondary route diagnosis，不是 GO gate。'))
    out.append(section(26, 'STRONG_RESIDUAL_DEEP_RECOVERY_SIGNAL',
        '`FALSE`。B/C/D 未通过，且两个 S_D 主 image AUC 分别0.5002、0.6083，均低于0.75。'
        '不能用 q 的0.9067或某个子群值宣称 Strong signal。'))
    out.append(section(27, 'Route attribution',
        table(['路径', '支持与限制', '本轮判断'], [
            ['A — missed Deep-Win', '113,204个机会；需理论恢复27.6192%，但 S_D 的utility/winner/cross-stratum门槛失败', '数量支持，预注册恢复排名不支持'],
            ['B — Both-Wrong context', '321,981个位置；ctx正确108,541个，条件救回33.7104%；one-correct污染尚未解决', '仅支持另行第三证据审计'],
        ]) + 'Residual Raw-Wrong 的多数人口是 Both-Wrong（约74%），但 beneficial 子集的多数仍是 missed Deep-Win（约64%）。'
        '“误差人口构成”与“有益机会构成”不可混为一谈。现有数据也不能证明所有冻结证据都无用：'
        '否定的是预注册 S_D 方案在本合同下的能力，不是对一切未来方案作不可能性证明。'))
    out.append(section(28, 'Scientific interpretation / engineering delivery',
        '本轮回答：残余覆盖率不足**不是没有潜在有益位置**，而是预注册的绝对 deep support '
        '尚不能可靠筛选它们；对大部分 neither-hierarchy-correct 的残余，邻域 context 有条件性的额外信息。\n\n'
        '这仍不足以安全修复 Gate E：有益标签是局部 logit-margin 方向导数，不等于有限更新的纠错；'
        '第三证据救回是在 GT 划分的人口上观察到的，尚缺 GT-blind 人口辨识与 one-correct 保护证据。'
        '没有得出“CH必然可以安全救回”或“直接用q训练”的结论。\n\n'
        '工程交付包含独立 tools、44项测试、26项独立复核、完整 CSV/JSON、10000 bootstrap replicates、'
        '输入SHA与运行记录、可复跑 README。原 `network/`、`tool/`、`train_sshr.py` 与 A0 完全一致。'
        '本轮使用既有缓存，不生成新的大特征文件或 checkpoint，不删除旧实验。\n\n'
        '[运行说明](README_rddr_phase2b110.md)；[交付摘要](rddr_phase2b110_delivery_summary.md)；'
        '[冻结合同](rddr_phase2b110_contract.md)。大输入缓存和 image_statistics NPZ 留服务器，'
        'CSV/JSON/报告进入独立分支供 PR 审核。没有新增训练、lambda/threshold搜索、恢复 gate 或新 loss。'))
    out.append(section(29, 'Exact final decision',
        '按已批准的 decision precedence：A PASS，B/C/D 非 PASS，第三证据信号 TRUE，故为第三证据路线支持。'
        '只允许在用户下一次确认后另立独立、预注册的 third-evidence rescue audit；'
        '**本轮到此停止，不自动设计机制，不训练，不做 Full25，不评 test/LUAD/其它seed。**'))
    out.append('DECISION = ' + s['decision'] + '\n')
    text = ''.join(out)
    assert sum(line.startswith('## ') for line in text.splitlines()) == 29
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or args.manifest.exists():
        raise FileExistsError('Refusing to overwrite delivery artifacts; choose new paths.')
    text = render(args.run)
    artifacts = sorted(p for p in args.run.iterdir() if p.is_file() and p.name.startswith(P)
                       and p.suffix in ('.json', '.csv', '.txt')
                       and p.name != P + 'delivery_manifest.json'
                       and p.resolve() != args.manifest.resolve())
    entries = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in artifacts}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)
    entries[args.report.name] = {'sha256': digest(args.report), 'bytes': args.report.stat().st_size}
    with args.manifest.open('x', encoding='utf-8', newline='\n') as f:
        json.dump({'schema': 1, 'files': entries, 'note': 'Hashes of immutable scientific evidence and generated report; manifest excludes itself.'},
                  f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'report': str(args.report), 'sha256': digest(args.report), 'manifest': str(args.manifest), 'files': len(entries)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
