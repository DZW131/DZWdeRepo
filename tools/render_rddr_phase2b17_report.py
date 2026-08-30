"""Render the complete frozen audit from verified artifacts; no model/data access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

P = 'rddr_phase2b17_'
LINK = '../audit/results/rddr_phase2b17/'


def number(value):
    s = str(value)
    try:
        x = float(s)
    except ValueError:
        return s.replace('|', '\\|')
    if x != x:
        return 'NA (无有效分母)'
    if x == int(x):
        return str(int(x))
    if 0 < abs(x) < .001:
        return f'{x:.8e}'
    return f'{x:.6f}'


def table(rows, columns):
    return '\n'.join([
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |',
        *['| ' + ' | '.join(number(r.get(c, '')) for c in columns) + ' |' for r in rows],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--manifest', type=Path)
    args = ap.parse_args()
    if args.output.exists() or (args.manifest and args.manifest.exists()):
        raise FileExistsError('Use a NEW output/manifest path; existing artifacts are preserved.')

    def js(name):
        return json.loads((args.results / (P + name + '.json')).read_text(encoding='utf-8'))

    def rows(name):
        with (args.results / (P + name + '.csv')).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    def data_table(name, columns, predicate=lambda r: True):
        file = P + name + '.csv'
        return table([r for r in rows(name) if predicate(r)], columns) + f'\n\n数据：[{file}]({LINK}{file})。'

    def code(value):
        return '```text\n' + value + '\n```'

    def command(value):
        return '```bash\n' + value.replace(' --', ' \\\n  --') + '\n```'

    s, rt, ver, identity, detach, smoke = [js(x) for x in (
        'summary', 'runtime', 'verification', 'identity_audit', 'detach_audit', 'bf16_smoke')]
    tests = (args.results / 'unit_integration_tests.txt').read_text(encoding='utf-8')
    assert 'Ran 29 tests' in tests and '\nOK\n' in tests and 'skipped=' not in tests
    assert ver['status'] == 'PASS' and len(ver['checks']) == 28 and all(ver['checks'].values())
    assert s['decision'] == 'CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED'
    assert [s['gate_' + k] for k in 'ABCD'] == ['FAIL'] * 4
    assert rt['optimizer_steps'] == 0 and not rt['test_access'] and not rt['checkpoint_written']
    parts = ['# RDDR Phase-2B1.7 Contextual Correction Acceptance Audit\n\n'
             '完整实验报告｜BCSS validation-only｜C0 Full25 seed42｜zero-update\n\n'
             '**结论：冻结的 contextual acceptance 方案未通过；A/B/C/D 全部 FAIL，工程验证 PASS。** '
             '接纳分数具有一定 winner 排序信号，但不足以安全选择教师纠正。'
             '本轮没有训练、没有 optimizer、没有 test/LUAD、没有阈值搜索，也没有改动或删除既有权重。\n\n'
             '所有 rate/accuracy/AUC 默认以 0–1 表示，pp 明确表示百分点；dM 是单位负梯度方向下的局部 logit margin 变化，'
             '不是 mIoU，不代表真实训练后的预测翻转率。表内显示值经过格式化，CSV/JSON 保留原始精度。']

    def section(index, title, *body):
        parts.append(f'## {index}. {title}\n\n' + '\n\n'.join(body))

    section(1, 'Provenance、SHA 与实际命令',
        f"纯 A0：`{rt['a0']}`。分支：`feature/rddr-phase2b17-acceptance`；PR 基线：`baseline/official-a0`。"
        f"实际 GPU 执行 commit：`{rt['code_commit']}`；最终独立复核 commit：`{ver['code_commit']}`。"
        '仅新增独立审计工具、测试、文档与结果；原网络、训练、推理、指标文件保持不变。',
        '合同：[rddr_phase2b17_contract.md](rddr_phase2b17_contract.md)。'
        '用户批准的规格为 `RDDR_Phase2B1_7_Contextual_Correction_Acceptance_Audit_v1.0.md`。',
        table([dict(asset=k, path=rt['paths'][k], SHA256=v) for k, v in rt['source_sha256'].items()],
              ['asset', 'path', 'SHA256']),
        'checkpoint 为 451,130,207 bytes；native/derived/previous 缓存分别为 '
        '240,845,567 / 301,590,961 / 155,185,711 bytes。新观察缓存位于 '
        '`/home/duyanhong/experiments/RDDR_PHASE2B17/formal_r1/rddr_phase2b17_observations.npz`，SHA256：'
        f"`{rt['observation_sha256']}`。大缓存不提交 Git。",
        f"环境：{rt['gpu']}；PyTorch `{rt['torch']}`；NumPy `{rt['numpy']}`；"
        f"{rt['precision']}；主审计 batch={rt['batch']}，补充 batch20。"
        '全部 3,418 张 validation，28×28 原生审计网格；2,479,143 有效前景位置分布在 3,416 张图。'
        '损失仍覆盖全部 2,679,712 原生位置，GT 不控制损失采样。',
        '实际 GPU 命令（已运行，不应覆盖原目录重跑）：', command(rt['command']),
        '实际统计与独立验证命令：', command(s['command']), command(ver['command']),
        '服务器工作目录：`/home/duyanhong/DZWdeRepo-rddr-phase2b17`。'
        '最终小型结果来自 `formal_r1` 与 `report_r3`，本地归档在 `audit/results/rddr_phase2b17/`。'
        '早期 `report_r1/r2` 均保留，数值验证修订过程见第25节。')

    section(2, '冻结的 Phase-2B1.6 证据',
        '上一轮为 `TEACHER_SIGNAL_PRESENT_GRADIENT_UNSAFE`，A/B/C/D = PASS/PASS/FAIL/PASS。'
        '以下是原生 CAM28_1 的冻结诊断，并非官方最终融合 segmentation 指标：',
        table([dict(model='rect', accuracy='81.7788%', mIoU='63.7895%'),
               dict(model='symmetric teacher', accuracy='78.5383%', mIoU='59.3171%')], ['model', 'accuracy', 'mIoU']),
        'Repair=88,290；Harm=168,626；NetRepair=-3.2405 pp。'
        'CCA all Benefit/Harm=17.9611%/82.0389%，Top20=31.2450%/68.7550%；'
        'Rect_Wrong Benefit=86.8583%，Rect_Correct Harm=97.3899%。'
        'q 的正权重只缩放逐像素梯度，不能反转方向。本轮检验的不是再次证明 teacher 有信号，'
        '而是能否接受有益纠正、拒绝有害纠正。')

    section(3, 'p_rect / p_teacher 张量与冻结重放',
        '`p_s/p_d/p_rect/p_teacher: [B,4,28,28]`，FP32；`q: [B,28,28]`。'
        'p_rect 从冻结 BF16 forward 产生的 FP32 logits 用 CUDA FP32 softmax 重建，'
        '不把上一轮统计时临时 FP64 softmax 当成冻结概率。teacher 仍是原 symmetric adjudication 的浅/深混合。',
        code('S_S_sym = 0.5*(T_SS+T_SD)\nS_D_sym = 0.5*(T_DS+T_DD)\n'
             'wD = S_D_sym/(S_S_sym+S_D_sym+1e-8)\np_teacher = (1-wD)*p_s+wD*p_d\n'
             'q = JS(p_s,p_d)/ln(2)'),
        table([dict(quantity=k, replay_max_abs=v) for k, v in rt['parity'].items()], ['quantity', 'replay_max_abs']),
        'teacher 与旧四项 support 完全一致；q 最大差 5.96046e-8，在预先批准的 1e-7 内，'
        '损失使用原缓存 q。U/CCA 损失与梯度完全一致；3,418 张真实网络 logits 重放 max_abs=0。'
        '因此“exact replay”不能误写成所有浮点中间量逐 bit 相等。')

    section(4, 'S_R / S_T：对称邻域支持',
        code('R_S(i)=mean_j[1-JS(p_rect(i),p_s(j))/ln2]\n'
             'R_D(i)=mean_j[1-JS(p_rect(i),p_d(j))/ln2]\n'
             'T_S(i)=mean_j[1-JS(p_teacher(i),p_s(j))/ln2]\n'
             'T_D(i)=mean_j[1-JS(p_teacher(i),p_d(j))/ln2]\n'
             'S_R=0.5*(R_S+R_D); S_T=0.5*(T_S+T_D)'),
        '15×15、radius=7、去掉中心自环、仅图内邻居。JS 保留旧实现 log 内 eps=1e-8。'
        '新 support 没有额外 clamp、归一化或 offset；GT 不进入构造。下表仅统计有效前景，计算本身覆盖全部位置。',
        data_table('support_rect_teacher', ['quantity', 'count', 'mean', 'std', 'median', 'min', 'max'],
                   lambda r: r['stratum'] == 'all' and r['quantity'] != 'delta'))

    section(5, 'Delta_accept：固定原始分数',
        code('Delta_accept = S_T-S_R\nm = (Delta_accept > 0)\na = relu(Delta_accept)'),
        data_table('support_rect_teacher', ['quantity', 'count', 'mean', 'std', 'median', 'min', 'max'],
                   lambda r: r['stratum'] == 'all' and r['quantity'] == 'delta'),
        '零阈值固定。有效前景中 Δ=0 有4个，全部拒绝。没有 offset、temperature、分数翻转、class rule 或阈值扫描。'
        '平均 S_T>S_R 不是 teacher 比 rect 更正确的证明；必须结合 winner 标签检验。')

    section(6, 'Teacher-Win / Rect-Win 诊断人群',
        '仅在 teacher 与 rect 预测不同且恰好一个正确的位置定义 winner 标签。'
        'Teacher-Win=88,290，Rect-Win=168,626，合计256,916，正类占比0.343653。'
        '其余有效前景并不进入 winner AUROC，但仍进入梯度/覆盖率统计。',
        '3,250张图存在 winner 样本；其中2,547张同时含两类，可计算 image AUROC。'
        '排除871张：168张无 winner 样本、703张仅单类。不能把排除图作为 AUC=0.5 填充。')

    section(7, 'Winner acceptance AUROC / AP / BA / recalls',
        data_table('acceptance_winner', ['stratum', 'positive', 'negative', 'auroc', 'image_auroc', 'auprc',
                                       'balanced_accuracy', 'macro_f1', 'teacher_win_recall', 'rect_win_recall'],
                   lambda r: r['stratum'] in ('all', 'Top20', 'Bottom80')),
        '主 image-AUROC=0.619465，95% CI [0.608924,0.630020]：高于随机，但未达到0.65。'
        '固定零阈值 BA=0.587359，TW recall=0.699139，RW recall=0.475579；'
        'BA和RW recall均未达门限。TP/FN/FP/TN=61,727/26,563/88,431/80,195。')

    section(8, 'Gradient-benefit AUROC',
        '标签复用冻结 CCA 的 exact first-order dM：正445,281，负2,033,862，dM=0为0。'
        '原始非GT最大 logit 存在2,698个并列位置，使用上一轮冻结的 tied-max directional derivative，未随意选单个竞争类。',
        data_table('gradient_discrimination', ['stratum', 'positive', 'negative', 'auroc', 'image_auroc', 'auprc', 'eligible_images'],
                   lambda r: r['stratum'] in ('all', 'Top20', 'Rect_Correct', 'Rect_Wrong',
                                              'class0', 'class1', 'class2', 'class3', 'boundary', 'interior')),
        '主 image-AUROC=0.523684，95% CI [0.516922,0.530530]，远未达到0.65。'
        '合格图2,889张；529张无双标签，其中2张无有效前景、527张仅一种梯度符号。')

    section(9, 'q 与 acceptance 是否提供不同证据',
        data_table('q_vs_acceptance', ['score', 'auroc', 'image_auroc', 'auprc', 'eligible_images']),
        '配对 image-AUROC 差 Δ−q=+0.002978，95% CI [-0.006820,+0.012930]，跨0。'
        'q（Need）与Δ（Trust）在定义上不同，但本轮不能声称Δ在 winner 判别上显著优于q。'
        '规格中的“证明是两个不同变量”不能取代实测证据；这里如实报告未建立更强判别能力。')

    section(10, 'Confidence controls（仅诊断）',
        data_table('confidence_controls', ['score', 'auroc', 'image_auroc', 'auprc']),
        '熵采用自然对数，JS控制保持未除ln2的原定义。所有方向预先固定，不对低于0.5的分数取负，'
        '不采用看起来更好的 maxconf 分数替代 primary Δ，不做任何融合。')

    section(11, 'Hard acceptance：primary consumption probe',
        code('L_HA = sum_i q_i*m_i*KL(p_teacher||p_rect) / (sum_i q_i*m_i+1e-8)'),
        '所有 teacher/q/Δ/m 均 detach。主实验每图 batch1，分母含全部784位置（包括背景与ignore），'
        'GT不用于选择损失位置。全拒绝图通过eps安全得到0损失和0梯度；本次真实图全拒绝数为0，'
        '人工全拒绝单测通过。未对完整 SSHR 分类目标做优化更新。')

    section(12, 'Soft acceptance：唯一 secondary probe',
        code('L_SA = sum_i q_i*relu(Delta_i)*KL(p_teacher||p_rect) / '
             '(sum_i q_i*relu(Delta_i)+1e-8)'),
        data_table('sa_gradient', ['stratum', 'benefit_rate', 'harm_rate', 'zero_rate', 'mean_dm', 'median_dm'],
                   lambda r: r['stratum'] in ('all', 'Top20', 'Rect_Correct', 'Rect_Wrong')),
        'SA与HA有相同接受位置与逐像素方向，只重分配幅度；未引入sigmoid、power或温度。'
        'SA all Mean_dM=-1.7945585e-4，较HA更接近零，但仍为负。')

    section(13, 'Acceptance rate 与分母',
        data_table('acceptance_population', ['stratum', 'targets', 'accepted', 'rejected', 'acceptance_rate', 'zero_delta']),
        '总体接受83.8245%，不存在“几乎全拒绝”的假安全。Top20与Q5分别沿用原冻结定义，'
        '本轮不重新排序或重建边界，因此两组大小不必相等。所有统计率默认整个有效前景stratum作分母。')

    section(14, 'Accepted / Rejected teacher quality',
        data_table('accepted_teacher_quality', ['stratum', 'region', 'targets', 'teacher_accuracy', 'rect_accuracy',
                                               'accuracy_delta', 'repair', 'harm', 'net_repair'],
                   lambda r: r['stratum'] in ('all', 'Top20')),
        'Accepted: teacher−rect=-1.2850 pp，95% CI [-1.5129,-1.0563] pp；'
        '净修复数=-26,704，95%图像bootstrap区间[-31,438.175,-21,953.875]。'
        'Rejected中teacher也较差（-13.3741 pp）。虽然拒绝区明显更差，接受区仍不满足teacher>rect。'
        'NetRepair_rate与accuracy_delta数学相等，不能视为两份独立成功证据。')

    section(15, 'Correction precision / recall / protection',
        data_table('selective_correction', ['stratum', 'Teacher_Win', 'Rect_Win', 'accepted_Teacher_Win',
                                           'accepted_Rect_Win', 'correction_precision', 'correction_recall', 'rect_protection_rate'],
                   lambda r: r['stratum'] in ('all', 'Top20', 'Bottom80')),
        'CorrectionPrecision只以接受的winner-conflict为分母，不等同teacher整体accuracy。'
        '总体接受的有用纠正61,727少于有害纠正88,431；对Rect-Win的保护率仅47.5579%。')

    gradcols = ['stratum', 'loss', 'benefit_rate', 'harm_rate', 'zero_rate', 'mean_dm', 'median_dm', 'active_fraction']
    section(16, 'HA 梯度审计',
        'g=dL/dlogits，下降方向v=-g。GT margin为 z_GT−max(z_nonGT)，dM沿v取精确一阶方向导数。'
        'Benefit/Harm/Zero互斥且以整个stratum计数，拒绝位置的0梯度必须保留。',
        data_table('ha_gradient', gradcols,
                   lambda r: r['stratum'] in ('all', 'Top20', 'Rect_Correct', 'Rect_Wrong', 'class0', 'class1', 'class2', 'class3')),
        'HA all Benefit14.2161% < Harm69.6084%；Top20 Benefit24.5982% < Harm49.5902%。'
        '接受条件下all Harm=83.0407%，仅作辅助，不替换主门限分母。')

    section(17, 'U / CCA / HA / SA 对照与 CCA→HA 主比较',
        data_table('all_gradient_controls', gradcols,
                   lambda r: r['stratum'] in ('all', 'Top20', 'Rect_Correct', 'Rect_Wrong') or
                   (r['stratum'] in ('class0', 'class1', 'class2', 'class3') and r['loss'] in ('CCA', 'HA'))),
        'HA−CCA all Mean_dM=+5.4979792e-5，CI[+5.1718891e-5,+5.8292067e-5]；'
        'all Harm下降12.4305 pp。说明拒绝机制确实减轻一部分伤害，但“负值变得不那么负”不等于安全。'
        '接受位置HA/SA没有反转CCA方向，拒绝位置梯度严格为0；相关恒等式及数值误差保存在 gradient_identities.json。')

    section(18, 'Rect-Correct protection',
        data_table('correct_wrong_safety', ['loss', 'targets', 'harm_rate', 'mean_dm', 'active_fraction'],
                   lambda r: r['stratum'] == 'Rect_Correct'),
        'CCA Harm=0.973899；HA/SA Harm=0.843256。固定要求HA≤0.486950（CCA的一半），未达到。'
        '绝对下降13.0643 pp，95% CI约[12.8471,13.2799] pp；不能把这一下降误读成减半。'
        '在已接受的Rect_Correct内，HA伤害率约98.7874%，表明本轮仍接收许多降低正确类别margin的教师方向。')

    section(19, 'Rect-Wrong correction',
        data_table('correct_wrong_safety', ['loss', 'targets', 'benefit_rate', 'harm_rate', 'mean_dm', 'active_fraction'],
                   lambda r: r['stratum'] == 'Rect_Wrong'),
        'HA Benefit=0.733741，95% CI[0.728212,0.739200]，达到0.60。'
        '由CCA的0.868583下降后仍保留纠错能力，故D失败不是“完全不纠错”，主要在正确student保护不足。')

    section(20, 'Gradient coverage',
        data_table('gradient_coverage', ['stratum', 'loss', 'targets', 'active_fraction', 'zero_rate', 'mean_gradient_norm'],
                   lambda r: r['loss'] in ('HA', 'SA') and
                   r['stratum'] in ('all', 'Top20', 'Bottom80', 'Rect_Correct', 'Rect_Wrong')),
        'all ActiveGradientFraction=0.838245，95% CI[0.836472,0.840069]，明显大于0.10。'
        '真实参数路径也进行了反传；未创建optimizer，不以是否nonzero取代科学门限。')

    section(21, 'q × acceptance 二维审计',
        data_table('q_acceptance_grid', ['quintile', 'acceptance', 'targets', 'accuracy_delta', 'net_repair', 'CCA_mean_dm', 'HA_mean_dm']),
        'Q1–Q5沿用冻结q分位边界。Reject格的HA dM为0，这是拒绝操作本身，不是有益方向。'
        '二维结果用于描述Need与Trust，不据此选择某个分位区间训练。')

    section(22, 'Per-class acceptance 与统计功效',
        data_table('per_class', ['stratum', 'Teacher_Win', 'Rect_Win', 'eligible_images', 'image_auroc', 'balanced_accuracy', 'power']),
        data_table('per_class', ['stratum', 'acceptance_rate', 'accepted_accuracy_delta', 'HA_benefit_rate', 'HA_harm_rate', 'HA_mean_dm']),
        '四类均满足≥500正类、≥500负类、≥30双标签图，不能用underpowered解释本次失败。'
        'class2 image-AUROC=0.403953，CI[0.378316,0.429693]，与class1的0.708264差异明显。'
        '所有四类HA Mean_dM仍为负。本轮不翻转class2、不排除类别、不创建class-specific规则。')

    section(23, 'Boundary / interior',
        '沿用冻结boundary≤7px、interior>7px标记；不改变计算尺度或重新定义边界。',
        data_table('boundary_interior', ['stratum', 'winner_image_auroc', 'winner_pooled_auroc', 'HA_benefit_rate',
                                        'HA_harm_rate', 'HA_mean_dm', 'accepted_accuracy_delta']),
        '边界image-AUROC=0.478573，未表现出可靠winner判别。边界HA Mean_dM虽略正，'
        '也不能取代主all/Top20/class门限，更不能只报告该子组作为成功。')

    section(24, '冻结 HFRM transition groups',
        '**历史名称by_CH实际为raw→完整HFRM transition，不是isolated CH因果分组。**',
        data_table('hfrm_groups', ['stratum', 'targets', 'acceptance_rate', 'mean_delta', 'teacher_rect_accuracy_delta', 'HA_mean_dm']),
        'Still_Wrong/Harmed_by_CH中的正向局部信号与Corrected_by_CH/Stable_Correct中的负向信号并存。'
        '这只能解释整体矛盾，不能据GT组别设计训练门控。')

    paramrows = rows('parameter_gradients')
    pg = []
    for mode in ('HA', 'SA'):
        for param in detach['allowed_parameters']:
            rr = [r for r in paramrows if r['loss'] == mode and r['parameter'] == param]
            assert len(rr) == 3418 and all(r['finite'] == 'True' for r in rr)
            pg.append(dict(loss=mode, parameter=param, images=len(rr),
                           nonzero_images=sum(float(r['max_abs']) > 0 for r in rr),
                           RMS_min=min(float(r['rms']) for r in rr), RMS_max=max(float(r['rms']) for r in rr)))
    section(25, 'Detach / identity / BF16 / 独立验证',
        '仅允许以下7个既有student参数张量求导；其余参数梯度始终None。'
        'teacher/q/Δ/m/a全部detach，语义来源分支与acceptance对rect的分支无梯度。'
        '共享ic1具有合法student梯度，不等于teacher分支漏梯度。', table(pg, ['loss', 'parameter', 'images', 'nonzero_images', 'RMS_min', 'RMS_max']),
        f"全部参数和BN buffer的state SHA前后相同：`{identity['state_before']}`。"
        f"checkpoint SHA前后相同。严格加载missing_keys={identity['missing_keys']}，unexpected_keys={identity['unexpected_keys']}。",
        '固定160图按32个等距索引+seed42抽取其余128图预先选定，未看结果挑图。'
        f"原官方推理、background overwrite前的8,028,160个预测像素SHA前后相同：`{identity['prediction_before']['prediction_sha256']}`。"
        'raw forward固定160也完全一致；全部3,418张logits与冻结Phase16重放完全一致。'
        '不得将160图prediction SHA测试宣称为3418图完整官方分割评估。',
        table([dict(batch=smoke['batch'], loss_HA=smoke['losses']['HA']['loss'], loss_SA=smoke['losses']['SA']['loss'],
                    seconds=smoke['seconds'], allocated_GiB=smoke['allocated_bytes']/1024**3,
                    reserved_GiB=smoke['reserved_bytes']/1024**3, budget_GiB=22, finite=smoke['all_finite'])],
              ['batch', 'loss_HA', 'loss_SA', 'seconds', 'allocated_GiB', 'reserved_GiB', 'budget_GiB', 'finite']),
        '上述显存仅限HFRM28_1/ic1选定参数反传，**不是全网络解冻Full25训练显存证明**。'
        f"主GPU审计总计{rt['total_seconds']:.3f}s（含缓存加载/前向/验证与产物）；"
        f"其中parity {rt['parity_seconds']:.3f}s、support {rt['support_seconds']:.3f}s、"
        f"实际逐图反传{rt['gradient_seconds']:.3f}s；统计分析{s['analysis_seconds']:.3f}s。"
        '这些是本次一次运行耗时，不作重复benchmark或端到端训练速度预测。',
        '**验证结果：29项unit/integration测试PASS、0skip；28项独立复核PASS。** '
        '独立验证不导入主实现/分析器，使用显式邻居gather、SciPy tie-rank AUROC/AP、'
        'FP64 epsilon-KL解析导数以及NumPy gather-sum图像bootstrap。',
        table([dict(check=k, passed=v) for k, v in ver['checks'].items()], ['check', 'passed']),
        table([dict(diagnostic=k, max_error=v) for k, v in ver['errors'].items()], ['diagnostic', 'max_error']),
        '**数值事件完整披露。** report_r1已写出统计后，console JSON打印遇到0d NumPy对象序列化错误；'
        '仅修复console序列化。report_r2的初版验证使用绝对误差≤2e-8对比FP64解析导数和FP32实测梯度，'
        '在SA大小0.583535的梯度上出现9.99457e-8差异（相对1.71e-7），因此该验证当时FAIL。'
        '随后增加全3418图独立同精度FP32 autograd精确重放（max_abs=0），以及FP64 autograd对解析式验证'
        '（max_abs=1.11022e-16），区分舍入与公式错误。未改变任何主观察数组、损失、接受分数、科学门限或结果。'
        '原失败记录保留在服务器report_r2，report_r3为最终通过的复核。',
        '独立support因求和顺序不同max_abs=3.57628e-7；共3个位置的Δ符号在FP32零附近不同，'
        '对应原|Δ|≤1.19209e-7。它满足预先实现的独立support 1e-6数值容差，但不是bitwise一致。'
        '主结果仍使用原固定Δ，不用独立重算值替换，也不增设“近零带”。'
        'bootstrap所有重复估计max差1.80444e-9，CI max差1.59707e-9。',
        '原始证据：[' + P + 'verification.json](' + LINK + P + 'verification.json)、'
        '[unit_integration_tests.txt](' + LINK + 'unit_integration_tests.txt)、'
        '[' + P + 'parameter_gradients.csv](' + LINK + P + 'parameter_gradients.csv)。')

    section(26, '10,000次配对图像bootstrap',
        'seed42；从全部3,418图像索引有放回抽样，10,000次，每次重新计算原估计量。'
        'image AUROC在该重复内的双标签图上等权；pooled rates用重新汇总的像素计数。'
        'HA−CCA与Δ−q配对，不做像素bootstrap。全部报告指标均10,000个有效重复。'
        '95%区间为percentile描述区间，未作多重检验校正；不据此搜索成功子组。',
        data_table('bootstrap', ['metric', 'estimate', 'ci_low', 'ci_high', 'valid_resamples']),
        f"随机抽样索引流SHA256：`{s['bootstrap_rng_sha256']}`。完整重复值保存在bootstrap_replicates.csv。")

    section(27, 'Gate A / B / C / D：逐条判决',
        table([
            dict(gate='A', condition='winner image-AUC ≥0.65', observed='.619465', result='FAIL'),
            dict(gate='A', condition='95% CI lower >0.50', observed='.608924', result='PASS'),
            dict(gate='A', condition='zero-sign BA ≥0.60', observed='.587359', result='FAIL'),
            dict(gate='A', condition='Teacher-Win recall ≥0.55', observed='.699139', result='PASS'),
            dict(gate='A', condition='Rect-Win recall ≥0.55', observed='.475579', result='FAIL'),
            dict(gate='B', condition='gradient image-AUC ≥0.65', observed='.523684', result='FAIL'),
            dict(gate='B', condition='95% CI lower >0.50', observed='.516922', result='PASS'),
            dict(gate='C', condition='HA all Benefit > Harm', observed='.142161 < .696084', result='FAIL'),
            dict(gate='C', condition='HA Top20 Benefit > Harm', observed='.245982 < .495902', result='FAIL'),
            dict(gate='C', condition='positive Mean_dM in ≥5/6 strata', observed='0/6; all classes sufficiently powered', result='FAIL'),
            dict(gate='D', condition='HA Rect_Correct Harm ≤0.5×CCA', observed='.843256 > .486950', result='FAIL'),
            dict(gate='D', condition='HA Rect_Wrong Benefit ≥0.60', observed='.733741', result='PASS'),
            dict(gate='D', condition='HA all ActiveGradientFraction ≥0.10', observed='.838245', result='PASS'),
        ], ['gate', 'condition', 'observed', 'result']),
        'A/B/C/D = **FAIL / FAIL / FAIL / FAIL**。Engineering=PASS。'
        '按批准的优先级，工程通过后A失败即给出CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED；其余失败仍全部披露。')

    section(28, 'SOFT_ACCEPTANCE_PROMISING',
        '`TRUE`。SA all Mean_dM=-0.000179455851 > HA=-0.000234511453；'
        'SA Rect_Correct Harm=HA=0.843255990，满足不超过HA+5pp。'
        '这是相对secondary标志，不是主Go；SA的Mean仍负，不能据此绕过A/B/C/D开启训练。')

    section(29, 'STRONG_ACCEPTANCE_SIGNAL',
        '`FALSE`。winner/gradient image-AUC均低于0.75；HA all和Top20 Mean_dM均非正；'
        'Rect_Correct Harm远高于0.25。虽然Rect_Wrong Benefit≥0.70，不能抵消其余必要条件失败。')

    section(30, '科学解释、局限与交付边界',
        '1. 冻结的context支持差具有一定winner排序信息，却不足以决定什么时候teacher优于当前rect。'
        '接受区teacher accuracy仍显著低于rect，说明“邻域更一致”不等于“更正确”。\n'
        '2. 局部梯度损害明显多于获益。HA通过拒绝减少伤害，但正确student保护不充分；'
        '它不能改变接受位置的教师梯度方向。\n'
        '3. 失败不是没有工程连通、NaN、空loss或几乎全拒绝：83.82%前景梯度活跃，真实BF16反传和所有复核通过。\n'
        '4. class2、边界与其他子群表现差异不能授权规则搜索；本轮不翻转类别或替换primary score。\n'
        '5. 结论针对当前冻结S_T−S_R、零阈值和HA消费规则，不证明所有contextual acceptance思路都不可能。\n'
        '6. dM仅是固定checkpoint的局部logit导数审计；没有参数步进、没有联合分类loss训练，'
        '因此不能把HarmRate解释为预测翻转率或断言Full25必然损失相同幅度的mIoU。',
        '本轮代码和报告按独立分支交付，PR不自动merge。复现命令与目录索引见 '
        '[README_rddr_phase2b17.md](README_rddr_phase2b17.md)，交付清单见 '
        '[rddr_phase2b17_delivery_summary.md](rddr_phase2b17_delivery_summary.md)。'
        '归档只增加新目录，不删除旧实验、不覆盖C0和前三轮缓存。审计到此停止，不进入Phase-2B2/Full25。')

    section(31, 'Exact decision',
        '主科学门限失败。不得据soft flag或局部子组擅自推进训练，也不提供本轮协议以外的优化补丁。',
        'DECISION = ' + s['decision'])

    report = '\n\n'.join(parts) + '\n'
    assert report.count('\n## ') == 31 and report.rstrip().endswith('DECISION = ' + s['decision'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='\n') as f:
        f.write(report)
    if args.manifest:
        artifact_rows = []
        for path in sorted(args.results.iterdir()):
            if path.is_file() and path != args.manifest:
                b = path.read_bytes()
                artifact_rows.append(dict(file=path.name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()))
        b = args.output.read_bytes()
        manifest = dict(artifacts=artifact_rows, report=dict(file=args.output.name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()),
                        source_sha256=rt['source_sha256'], observation_sha256=rt['observation_sha256'],
                        note='Manifest excludes itself. Large observation NPZ is server-only. Files preserved byte-for-byte.')
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open('w', encoding='utf-8', newline='\n') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps(dict(report=str(args.output), bytes=len(report.encode('utf-8')), sections=31,
                          sha256=hashlib.sha256(report.encode('utf-8')).hexdigest()), ensure_ascii=False))


if __name__ == '__main__':
    main()
