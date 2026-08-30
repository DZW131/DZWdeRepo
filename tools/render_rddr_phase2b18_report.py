"""Complete 38-section report from validated artifacts. Standard library only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
P='rddr_phase2b18_';LINK='../audit/results/rddr_phase2b18/'


def fmt(x):
    try:v=float(x)
    except (ValueError,TypeError):return str(x).replace('|','\\|')
    if v!=v:return 'NA（无对应人群）'
    if v==int(v):return str(int(v))
    return f'{v:.8e}' if 0<abs(v)<.001 else f'{v:.6f}'


def table(rows,cols):
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |',
                      *['| '+' | '.join(fmt(r.get(c,'')) for c in cols)+' |' for r in rows]])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--manifest',type=Path)
    args=ap.parse_args()
    if args.output.exists() or (args.manifest and args.manifest.exists()):raise FileExistsError('Preserve old outputs; use a new path.')
    def js(name):return json.loads((args.results/(P+name+'.json')).read_text(encoding='utf-8'))
    def rows(name):
        with (args.results/(P+name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    def tab(name,cols,where=lambda r:True):
        file=P+name+'.csv'
        return table([r for r in rows(name) if where(r)],cols)+f'\n\n数据：[{file}]({LINK}{file})。'
    def cmd(s):return '```bash\n'+s.replace(' --',' \\\n  --')+'\n```'
    def code(s):return '```text\n'+s+'\n```'
    s,rt,v,ident,dt,smoke=[js(x) for x in ('summary','runtime','verification','identity_audit','detach_audit','bf16_smoke')]
    tests=(args.results/'unit_integration_tests.txt').read_text(encoding='utf-8')
    assert 'Ran 37 tests' in tests and '\nOK\n' in tests and 'skipped=' not in tests
    assert v['status']=='PASS' and len(v['checks'])==29 and all(v['checks'].values())
    assert [s['gate_'+k] for k in 'ABCDE']==['PASS','PASS','FAIL','FAIL','PASS']
    assert s['decision']=='TEACHER_BETTER_THAN_RAW_BUT_GRADIENT_UNSAFE'
    text=['# RDDR Phase-2B1.8 Pre-Rectification Teacher Guidance & Hierarchy-Safety Audit\n\n'
          '完整实验报告｜BCSS validation 3418张｜C0 Full25 seed42｜零更新、零搜索\n\n'
          '**结论：A/B/C/D/E = PASS / PASS / FAIL / FAIL / PASS。整体语义收益成立，但错误像素纠正能力与层级安全未达标，不进入 Full25。**\n\n'
          '本轮的进展是：teacher→raw 的 all/Top20 梯度收益均为正，四类Mean_dM也全部为正。'
          '主要风险是 Shallow-Win（浅层正确、深层错误）HHCR=96.2258%，teacher在该组硬标签正确率仅37.2862%。'
          'Raw-Wrong BenefitRate=60.6942%，低于70%门限。\n\n'
          '这不是一次训练结果：optimizer构建/step均为0，checkpoint、BN、原推理不变；未访问test/LUAD。'
          'native28 teacher mIoU提升不能当作最终融合分割mIoU的训练提升。所有rate/AUC默认0–1，pp为百分点；'
          'dM/dQ为单位步长负梯度的局部导数，不是预测翻转率或真实长期collapse发生率。']
    def sec(i,title,*body):text.append(f'## {i}. {title}\n\n'+'\n\n'.join(body))
    gc=['stratum','loss','targets','benefit_rate','harm_rate','zero_rate','mean_dm','median_dm']
    hc=['stratum','dQ_mean','dQ_median','dQ_negative_fraction','dQ_positive_fraction']
    pc=['stratum','teacher_accuracy','repair','harm','net_repair','benefit_rate','harm_rate','mean_dm','mean_dQ']

    sec(1,'Provenance / SHA / commands',
        f"纯A0：`{rt['a0']}`；分支`feature/rddr-phase2b18-prerect-guidance`，PR目标`baseline/official-a0`。"
        f"实际GPU执行commit：`{rt['code_commit']}`；独立复核commit：`{v['code_commit']}`。"
        '原网络、训练、推理、metric源码未改，仅新增tools/tests/docs/results。',
        '用户批准的[执行合同](rddr_phase2b18_contract.md)在新结果产生前提交。'
        '输入规格SHA256：`cc4f8588fdf04962ed447511144f6d03c2ffb7a6c7185b78c4ff5ac6b11553ca`。',
        table([dict(asset=k,path=rt['paths'][k],SHA256=h) for k,h in rt['source_sha256'].items()],['asset','path','SHA256']),
        f"新观察NPZ SHA256：`{rt['observation_sha256']}`，保留在"
        '`/home/duyanhong/experiments/RDDR_PHASE2B18/formal_r1/rddr_phase2b18_observations.npz`。'
        '不上传权重或大缓存到Git。正式统计与验证目录为`/home/duyanhong/experiments/RDDR_PHASE2B18/report_r1`。',
        f"环境：{rt['gpu']}，PyTorch `{rt['torch']}`，NumPy `{rt['numpy']}`；{rt['precision']}。"
        '原环境未升级。工作目录`/home/duyanhong/DZWdeRepo-rddr-phase2b18`。',
        '已执行的实际命令：',cmd(rt['command']),cmd(s['command']),cmd(v['command']))

    sec(2,'Frozen historical evidence',
        'Phase2B1.5 symmetric contextual ranking相对旧实现：image-AUROC 0.734850→0.784842，'
        'BA 0.593973→0.715627。Sym teacher相对FixedAvg native28 mIoU +1.9651pp，'
        '95% CI[+1.8105,+2.1300]pp。',
        'Phase2B1.6 teacher→rect无条件KL为`TEACHER_SIGNAL_PRESENT_GRADIENT_UNSAFE`；'
        'Phase2B1.7 acceptance为`CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED`。这两条路线保持No-Go，'
        '本轮未恢复post-HFRM KL、Δacceptance、feature disposal或receiver suppression。')

    sec(3,'Raw / teacher / rect 的相对位置',
        table([dict(prediction='raw',accuracy='.7142532721993043',native28_mIoU='.4363486817386499'),
               dict(prediction='symmetric teacher',accuracy='.7853834974424629',native28_mIoU='.5931706913402506'),
               dict(prediction='post-HFRM rect (frozen Phase16)',accuracy='.817788',native28_mIoU='.637895')],
              ['prediction','accuracy','native28_mIoU']),
        'raw < teacher < rect；因此移动监督位置有明确动机，但“teacher比raw强”不自动证明每种纠正都安全。'
        'rect行是既有诊断参考，本轮没有额外rect-guidance probe，也没有重新跑官方全量最终融合分割指标。')

    sec(4,'Tensor contract 与冻结重放',
        code('F28_raw: [B,512,28,28]\nL_s_raw, p_s: [B,4,28,28]\nDdeep: [B,4096,28,28]\nL_d, p_d, p_teacher: [B,4,28,28]\nq: [B,28,28]'),
        table([dict(quantity=k,max_abs=v) for k,v in rt['parity'].items()],['quantity','max_abs']),
        '旧native缓存没有raw logits。本轮从真实网络新提取raw logits，验证原ic1与frozen-head两条路径完全一致；'
        'softmax得到的ps/pd与旧缓存完全一致，teacher完全一致。q重算最大差5.96046e-8，'
        '在预注册1e-7容差内；loss使用冻结缓存q。未使用log(ps)反造logits。')

    sec(5,'Frozen-head student formulation',
        code('F28_raw = ReLU(bn45(b4_5(...b4(feat56))))\n'
             'L_s_student = conv2d(F28_raw, ic1.weight.detach(), ic1.bias.detach())'),
        '读取原HFRM28_1输入，不使用rectified输出作为student。39个批准的上游参数张量临时允许求导：'
        'b4/b4_1…b4_5以及bn45（含其BN affine）。所有BN仍eval、buffer不变。'
        '该临时梯度审计不改变原训练的BN冻结政策；b3及更早、b5及更深、全部HFRM和primary ic1均不接收梯度。')

    sec(6,'PRG loss',
        code('L_PRG = sum_i q_i KL(p_teacher_i || softmax(L_s_student)_i) / (sum_i q_i + 1e-8)'),
        'teacher、q、deep source全部detach；q仅用于强调supervision，不作为接受器或正确性选择器。'
        'KL逐类采用 t*(log(t+1e-8)−log(p+1e-8))。系数1；无lambda、threshold、temperature或q指数搜索。'
        '主审计batch1、每图全部784位置作为loss分母，包括背景/ignore；GT只进入事后诊断。')

    sec(7,'Uraw / FAraw controls',
        code('Uraw = mean_i KL(sym_teacher || p_s_student)\n'
             'FixedAvg = stopgrad(0.5*p_s + 0.5*p_d)\n'
             'FAraw = sum_i q_i KL(FixedAvg || p_s_student)/(sum_i q_i+eps)'),
        tab('raw_gradient',gc,lambda r:r['stratum'] in ('all','Top20','Raw_Correct','Raw_Wrong')),
        '仅上述三种训练时probe；shared-head是同一PRG的梯度路径诊断，不是第四个teacher或另一个损失方案。')

    sec(8,'Raw vs teacher semantic metrics',
        tab('teacher_raw_metrics',['model','accuracy','miou','dice','nll','brier','iou_class0','iou_class1','iou_class2','iou_class3']),
        'GT0–3前景的native28 pooled4×4混淆矩阵；bg4/ignore255排除，不用官方background overwrite；'
        'union为0的类别标NA并从宏平均排除。NLL=-log(p_GT+eps)，Brier为四类平方误差之和，不除4。'
        '实际有效前景2,479,143个，分布于3,416张图；其余2张仍参与无GT损失与身份检查。',
        'Teacher−raw accuracy=+7.1130pp，95% CI[+6.6530,+7.5727]pp；'
        'mIoU=+15.6822pp，95% CI[+14.8152,+16.5384]pp。历史raw/teacher完整精度指标复算一致。')

    sec(9,'Repair / Harm / NetRepair',
        tab('teacher_raw_transition',['stratum','targets','repair','harm','net_repair','net_repair_rate']),
        '总体Repair291,045，Harm114,703，NetRepair+176,342（95% CI[164,899.8,187,869.025]）。'
        'NetRepair_rate=teacher−raw accuracy，是同一个估计量，不是第二份独立证据。')

    sec(10,'TeacherAdvRaw',
        code('TeacherAdvRaw_i = p_teacher(GT)-p_raw(GT)'),
        tab('teacher_advantage',['stratum','targets','mean','median','positive_fraction','negative_fraction','zero_fraction']),
        'GT只定义观察量，绝不用于损失、teacher或训练样本过滤。')

    sec(11,'Raw GT-margin gradient',
        code('g_s=dL/dL_s_student\nv=-g_s  # unit step, NOT unit-normalized vector\n'
             'dM=v_GT - max(v_k for k tied at CURRENT maximal non-GT logit)'),
        f"冻结非GT最大logit并列位置{s['raw_tied_competitor_pixels']}个；PRG dM=0有{s['gradient_zero_pixels']}个。"
        '使用精确tied-max方向导数，不随意取某个并列竞争类。严格>0/<0符号，不增设近零阈值。'
        'logit/q梯度FP32，dM、dQ、norm和统计FP64累加。')

    sec(12,'Primary Benefit / Harm / Mean_dM',
        tab('raw_gradient',gc,lambda r:r['loss']=='PRG'),
        'all Benefit78.9280% > Harm21.0718%；Top20 Benefit71.9677% > Harm28.0313%。'
        'all、Top20与四类共6/6组Mean_dM>0，超过所需5/6，Gate B通过。'
        '分母始终为该stratum全部有效前景，零dM不剔除。')

    sec(13,'Raw-Correct protection',
        tab('correct_wrong_safety',pc,lambda r:r['stratum']=='Raw_Correct'),
        'Raw-Correct PRG HarmRate=13.7772%，95% CI[13.3821,14.1790]%，满足≤50%。'
        '该总体保护指标不错，但会掩盖Shallow-Win子群的严重风险，不能替代层级安全门限。')

    sec(14,'Raw-Wrong correction',
        tab('correct_wrong_safety',pc,lambda r:r['stratum']=='Raw_Wrong'),
        'Raw-Wrong BenefitRate=60.6942%，95% CI[59.7997,61.5708]%，不足70%。'
        '因此Gate C失败来自错误像素的纠正能力不足，而非Raw-Correct总体伤害率超标。')

    sec(15,'Deep-Win',
        tab('deep_shallow_win',pc,lambda r:r['stratum']=='Deep-Win'),
        '314,730个位置；teacher正确率92.0179%，PRG BenefitRate100%，Mean_dM=+0.00200433。'
        '当deep正确、raw错误时，这种指导几乎总能提高GT margin。')

    sec(16,'Shallow-Win',
        tab('deep_shallow_win',pc,lambda r:r['stratum']=='Shallow-Win'),
        '182,899个位置；teacher正确率37.2862%，hard-label harm62.7138%；'
        'PRG HarmRate96.2269%，Mean_dM=-0.00121071。浅层独有的正确信息没有得到保护。',
        '此处teacher硬标签正确率不等同Phase2B1.5 support-sign的Shallow-Win recall=79.0939%。'
        '前者评价混合概率的argmax，后者评价support分数选择方向；不能因之前ranking较好就假定混合teacher会保留浅层标签。')

    sec(17,'dQ directional derivative',
        code('g_q = d[JS(softmax(L_s), stopgrad(p_d))/ln2]/dL_s\ndQ = sum_k g_q[k]*(-g_s[k])'),
        'q导数在独立图计算，不回流到PRG的q权重。逐像素q先求和再对logits求导，得到block-diagonal逐像素导数，'
        '不是对图像均值求导造成的额外1/784缩放。',
        tab('hierarchy_direction',hc,lambda r:r['loss']=='PRG'),
        '总体99.9987%位置dQ<0：PRG几乎普遍削弱浅深冲突。冲突降低本身既不等于语义改善，也不等于有害collapse。')

    sec(18,'CosCollapse',
        code('CosCollapse = dQ / (norm(g_q)*norm(v)+1e-8)'),
        tab('hierarchy_direction',['stratum','CosCollapse_mean','CosCollapse_median','CosCollapse_negative_fraction','CosCollapse_positive_fraction'],lambda r:r['loss']=='PRG'),
        'all均值-0.870454，Top20均值-0.994792。保留原eps，未为小梯度调整分母；'
        '特别是Q1 norm乘积很小时，eps会减小绝对值，因此不能把该量一概视为无稳定项的标准余弦。')

    sec(19,'Beneficial Reconciliation Rate（BRR）',
        'BRR = Deep-Win内部 fraction(dM>0 AND dQ<0)，分母包含该组全部位置。',
        tab('reconciliation_collapse',['stratum','Deep_Win_targets','BRR'],lambda r:r['stratum'] in ('all','Top20','Bottom80','boundary','interior')),
        '总体BRR=99.9994%，95% CI[99.9984,100]%，通过≥60%要求。')

    sec(20,'Harmful Hierarchy Collapse Rate（HHCR）',
        'HHCR = Shallow-Win内部 fraction(dM<0 AND dQ<0)，不是总体像素中的比例。',
        tab('reconciliation_collapse',['stratum','Shallow_Win_targets','HHCR'],lambda r:r['stratum'] in ('all','Top20','Bottom80','boundary','interior')),
        '总体HHCR=96.2258%，95% CI[96.0071,96.4334]%，远高于≤30%要求。'
        '这是已预注册的局部方向风险事件，不是实际运行Full25后测得96%的信息消失或标签翻转。')

    sec(21,'Stable-Correct diversity',
        tab('deep_shallow_win',pc,lambda r:r['stratum']=='Stable-Correct'),
        tab('hierarchy_direction',['stratum','dQ_mean','CosCollapse_mean','dQ_negative_fraction'],lambda r:r['stratum']=='Stable-Correct' and r['loss']=='PRG'),
        '两路都正确的1,587,837个位置，teacher hard-label准确率仍100%；PRG Benefit95.7199%，'
        'Mean_dM正、dQ负。这类协调与Shallow-Win的有害冲突压缩必须分开解释。')

    sec(22,'Both-Wrong diagnostic',
        tab('deep_shallow_win',pc,lambda r:r['stratum']=='Both-Wrong'),
        'Both-Wrong共393,677；teacher仅纠正1,437个，正确率0.3650%。'
        'PRG Benefit29.2707%、Harm70.7283%、Mean_dM负；它拉低了Raw-Wrong总体纠正率。'
        '这是secondary观察，不引入third-evidence branch或依据GT挑选训练组。')

    sec(23,'Conflict localization',
        tab('gradient_localization',['loss','Top20_mean_G','Bottom80_mean_G','ratio']),
        'PRG Top20/Bottom80梯度范数比6.83507，高于Uraw3.38753，定位标志为TRUE。'
        '这只证明q更强调高冲突位置，不证明这些位置的teacher一定正确。')

    sec(24,'q quintiles',
        tab('q_quintiles',['stratum','targets','net_repair','benefit_rate','harm_rate','mean_dm','mean_gradient_norm','mean_dQ']),
        'Q5 mean G=0.00151134 > Q1=1.39841e-6。Top20/Q1–Q5沿用旧缓存和冻结边界，'
        'Top20与Q5不是重新按同一规则选取的等同集合。本轮不只用Q5/Top20训练。')

    sec(25,'Per-class semantic safety',tab('per_class',pc),
        '四类Mean_dM均正，且bootstrap下界均正；但类内总体改善不能抵消跨层winner子群的风险。'
        '未创建class-specific阈值或掩码。')

    sec(26,'Boundary / interior',
        '复用冻结boundary≤7px、interior>7px标签，未重算尺度或改边界宽度。',
        tab('boundary_interior',['stratum','targets','net_repair','benefit_rate','harm_rate','mean_dm','BRR','HHCR']),
        'BRR/HHCR以对应boundary/interior内的Deep-Win/Shallow-Win作分母。无相应人群则NA，不写成0。')

    sec(27,'Feature gradient',
        tab('feature_gradient',['stratum','targets','channels','rms','mean_pixel_l2','max_abs','finite']),
        '来自真实frozen-head PRG对F28_raw的反传。RMS跨该组所有像素×512通道，mean pixelL2先跨通道再平均。'
        '完整特征梯度未落盘；逐像素平方和和最大绝对值以小缓存保存，避免多GB冗余。')

    sec(28,'Upstream parameter gradient',
        tab('parameter_gradient',['parameter','numel','rms','max_abs','nonzero_images','zero_images'],lambda r:r['loss']=='PRG_frozen'),
        '所有39个批准上游参数均逐图检查finite；每个b4 block的卷积组汇总能量非零。'
        'primary ic1权重/偏置均无梯度；HFRM、deep-only路径及其余未批准参数均无梯度。'
        'BN affine仅临时用于诊断，running statistics从未更新。')

    sec(29,'Shared-head absorption diagnostic',
        tab('shared_head_diagnostic',['mode','feature_l2','ic1_l2','upstream_l2','head_parameter_energy_fraction','head_fraction_per_image_mean','head_fraction_per_image_median']),
        '各能量先逐图求梯度平方和，再跨图相加；分母仅ic1+批准上游参数，不混入feature-gradient能量。'
        '共享头占比0.99212%，远低于50%，吸收风险标志FALSE。frozen/shared特征梯度逐元素max差0，'
        '上游梯度总能量相同。本轮没有证据支持“监督主要被分类头吸收”。',
        '该比例依赖参数化与所开放的上游集合，不是未来optimizer更新分配的因果测量，不能单凭它主张解冻head。')

    sec(30,'Detach tests 与独立数值验证',
        '**37项unit/integration测试全部PASS，0skip；29项独立检查全部PASS。**',
        table([dict(check=k,passed=v) for k,v in v['checks'].items()],['check','passed']),
        table([dict(diagnostic=k,max_error=x) for k,x in v['errors'].items()],['diagnostic','max_error']),
        '独立验证器不导入主实现/分析器：重放三种FP32损失与logit梯度、独立q导数，'
        '用FP64解析epsilon-KL/JS导数对autograd验证，再独立重算margin/层级/混淆矩阵和bootstrap。'
        'FP32 loss、g_s、g_q误差均0；FP64 KL/JS解析差分别约6.94e-18/1.94e-16。'
        '本轮没有因为验证失败而调整阈值、容差或实验公式。',
        f"[verification.json]({LINK}{P}verification.json)、[测试原始记录]({LINK}unit_integration_tests.txt)、"
        f"[detach记录]({LINK}{P}detach_audit.json)。")

    sec(31,'Batch20 BF16 backward smoke 与资源',
        table([dict(batch=20,PRG_loss=smoke['loss'],seconds=smoke['seconds'],allocated_GiB=smoke['allocated_bytes']/1024**3,
                    reserved_GiB=smoke['reserved_bytes']/1024**3,conv_energy=smoke['upstream_conv_energy'],head_energy=smoke['head_energy'],passed=smoke['pass'])],
              ['batch','PRG_loss','seconds','allocated_GiB','reserved_GiB','conv_energy','head_energy','passed']),
        '使用固定32张的前20张，真实BF16网络/反传、FP32 loss、整个batch分母。teacher/q/deep来源梯度断开。'
        '预算22GiB；这是b4段临时反传显存，不证明Full25全网络解冻显存。',
        f"GPU主审计总计{rt['total_seconds']:.3f}s，冻结重放阶段{rt['parity_seconds']:.3f}s，"
        f"真实上游反传{rt['gradient_seconds']:.3f}s；统计分析{s['analysis_seconds']:.3f}s。"
        '以上为本次记录，不是重复benchmark或训练耗时预测。')

    sec(32,'Zero-update / inference identity',
        f"state_dict和BN buffer前后SHA一致：`{ident['state_before']}`。checkpoint SHA前后相同，"
        f"严格加载missing_keys={ident['missing_keys']}，unexpected_keys={ident['unexpected_keys']}。",
        '固定32个等距索引+seed42无放回抽取其余128图；未看结果挑图。原官方infer路径前后160图、'
        f"8,028,160个预测像素SHA：`{ident['prediction_before']['sha256']}`，在background overwrite之前计算。"
        '固定160图raw logits前后完全相同，全3418图raw梯度路径logits与无梯度路径完全相同。',
        '未创建optimizer、未step、未写checkpoint；没有test/LUAD/train split访问、没有新种子、没有完整训练。'
        '原推理仍仅运行A0，无teacher/support/q-loss额外计算；本轮没有新增inference FLOPs。')

    sec(33,'10,000次配对图像bootstrap',
        'seed42，从全部3418图有放回抽样。每次重汇总混淆矩阵和分子/分母，'
        '不是平均image mIoU，也不是pixel bootstrap。全部19个指标均10000个有效重复；'
        '报告percentile95% CI，未做多重检验校正，不据CI挑选子组或更改gate。',
        tab('bootstrap',['metric','estimate','ci_low','ci_high','valid_resamples']),
        f"抽样索引流SHA256：`{s['bootstrap_rng_sha256']}`。完整重复值已归档；独立gather复算误差约2.28e-18。")

    sec(34,'Gate A / B / C / D / E',
        table([
            dict(gate='A',requirement='teacher accuracy/mIoU > raw; NetRepair>0; CI lower>0',observed='+7.1130pp / +15.6822pp; +176342; both CI positive',result='PASS'),
            dict(gate='B',requirement='all and Top20 Benefit>Harm',observed='78.9280>21.0718%; 71.9677>28.0313%',result='PASS'),
            dict(gate='B',requirement='Mean_dM>0 in at least5/6 groups',observed='6/6',result='PASS'),
            dict(gate='C',requirement='Raw_Wrong Benefit≥70%',observed='60.6942%',result='FAIL'),
            dict(gate='C',requirement='Raw_Correct Harm≤50%; NetRepair>0',observed='13.7772%; +176342',result='PASS'),
            dict(gate='D',requirement='Deep-Win BRR≥60%',observed='99.9994%',result='PASS'),
            dict(gate='D',requirement='Shallow-Win HHCR≤30%',observed='96.2258%',result='FAIL'),
            dict(gate='D',requirement='Shallow-Win teacher accuracy≥60%',observed='37.2862%',result='FAIL'),
            dict(gate='D',requirement='Q5 mean G>Q1',observed='.00151134 > .00000139841',result='PASS'),
            dict(gate='E',requirement='finite/detach/upstream/batch20/identity/no-step',observed='all verified',result='PASS'),
        ],['gate','requirement','observed','result']),
        '汇总A/B/C/D/E=PASS/PASS/FAIL/FAIL/PASS。按批准优先级，E通过、A通过、C失败，'
        '最终标签为TEACHER_BETTER_THAN_RAW_BUT_GRADIENT_UNSAFE；D的层级风险也完整报告，不因决策优先级隐藏。')

    sec(35,'Secondary flags',
        'CONFLICT_LOCALIZATION_CONFIRMED=TRUE（PRG比Uraw更集中高冲突区域）。\n\n'
        'SHARED_HEAD_ABSORPTION_RISK=FALSE（head平方能量占比0.99212%≤50%）。\n\n'
        '两项都不覆盖或放宽C/D主门限，不能单独解锁训练。')

    sec(36,'STRONG_PRERECT_GUIDANCE_SIGNAL',
        'FALSE。teacher−raw mIoU≥10pp、all/Top20 Mean_dM正、Raw-Correct Harm≤35%、BRR≥70%均满足；'
        '但Raw-Wrong Benefit60.6942%未达到80%，HHCR96.2258%远高于20%。不得只列通过部分作为Strong Go。')

    sec(37,'Scientific interpretation 与局限',
        '**整体有效，但并非层级安全。** 与teacher→rect不同，teacher→raw的全局梯度收益确实成立。'
        '失败不是工程断梯度或分类头吸收，而是两类结构性问题：Both-Wrong纠正有限、Shallow-Win有效分歧被压低。',
        '**为何contextual teacher仍可能变成向deep靠拢？** 在当前冻结点，teacher=(1−wD)·p_s+wD·p_d。'
        '忽略log内极小eps项时，softmax-KL对raw logits的导数可直接化简为：',
        code('g_PRG ≈ [q/(sum(q)+eps)] * wD * (p_s-p_d)\n'
             'v_PRG ≈ positive_scalar * (p_d-p_s)'),
        '这是公式推导，不是新加一个deep-teacher实验。它表明在该冻结点，context的wD主要调节“向deep靠拢”的幅度，'
        '并不因Shallow-Win而自动反转方向。实际eps-KL梯度已精确计算和复核；观测上几乎所有位置dQ<0，'
        '因此有益reconciliation与有害dissent抹除同时出现。没有将近似等式伪称为含eps实现逐bit恒等式。',
        'Shallow-Win占所有Raw-Correct的一部分，所以Raw-Correct总体Harm仅13.78%并不能排除其子群96.23%的风险。'
        '另外，support-sign偏好浅层也不保证混合分布argmax仍选浅层类别。',
        '本轮仅固定checkpoint、单位局部下降方向，没有参数更新、没有联合SSHR分类损失、没有lambda和长期trajectory。'
        '因此不能宣称Full25已经塌缩，也不能把native28 teacher的+15.68pp当作最终模型mIoU提升。'
        '39个临时上游参数包含BN affine，属于审计白名单，不授权改变未来正式训练的BN冻结。',
        '交付包括完整CSV/JSON、测试、SHA清单、[复现README](README_rddr_phase2b18.md)和'
        '[交付摘要](rddr_phase2b18_delivery_summary.md)。独立PR不自动merge；旧实验/缓存保持不动。'
        '不追加阈值、lambda、GT保护掩码、third-evidence或新架构补丁。')

    sec(38,'Exact final decision',
        '预注册C/D门限未通过。完成本轮报告后停止，不进入Phase-2B2、Full25或test。',
        'DECISION = '+s['decision'])
    report='\n\n'.join(text)+'\n';assert report.count('\n## ')==38
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='\n') as f:f.write(report)
    if args.manifest:
        artifacts=[]
        for path in sorted(args.results.iterdir()):
            if path.is_file() and path!=args.manifest:
                b=path.read_bytes();artifacts.append(dict(file=path.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        b=args.output.read_bytes();manifest=dict(artifacts=artifacts,report=dict(file=args.output.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),
            observation_sha256=rt['observation_sha256'],source_sha256=rt['source_sha256'],note='Manifest excludes itself; large caches remain server-only.')
        with args.manifest.open('w',encoding='utf-8',newline='\n') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(args.output),sections=38,bytes=len(report.encode()),sha256=hashlib.sha256(report.encode()).hexdigest()),ensure_ascii=False))


if __name__=='__main__':main()
