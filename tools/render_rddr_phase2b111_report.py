"""Render the approved Phase111 report from verified evidence. Stdlib only, no overwrite."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

P='rddr_phase2b111_'
def num(x,d=4):
    if x is None or x=='':return 'NA'
    v=float(x);return f'{v:.{d}f}' if math.isfinite(v) else 'NA'
def pct(x):
    return num(float(x)*100)+'%' if x is not None and x!='' and math.isfinite(float(x)) else 'NA'
def cnt(x):return f'{int(float(x)):,}'
def table(headers,rows):
    def line(r):return '| '+' | '.join(str(v).replace('|','\\|') for v in r)+' |\n'
    return line(headers)+line(['---']*len(headers))+''.join(line(r) for r in rows)+'\n'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def render(run):
    def js(k):return json.loads((run/(P+k+'.json')).read_text(encoding='utf-8'))
    def rows(k):
        with (run/(P+k+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    s,rt,v=(js(k) for k in ('summary','runtime','verification'))
    tests=(run/(P+'tests.txt')).read_text(encoding='utf-8')
    assert v['status']=='PASS' and all(v['checks'].values()) and 'Ran 54 tests' in tests and tests.rstrip().endswith('OK')
    assert s['decision']==v['decision']=='THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT'
    assert [s['gate_'+k] for k in 'ABCDEF']==['FAIL','FAIL','PASS','PASS','PASS','FAIL']
    cirows=rows('bootstrap');cis={r['metric']:r for r in cirows}
    def ci(key,rate=False):
        r=cis[key];fmt=pct if rate else num;return '['+fmt(r['ci_low'])+', '+fmt(r['ci_high'])+']'
    def sec(n,title,body):return f'## {n}. {title}\n\n{body.rstrip()}\n\n'
    def source(file):return f'[完整 CSV](../audit/results/rddr_phase2b111/{P}{file}.csv)\n\n'
    h=s['headroom'];rp=rt['replay'];a=s['primary']['rescue'];b=s['primary']['bothwrong'];g=s['primary']['gradient'];gu=s['candidate_gradient']
    out=['# RDDR Phase-2B1.11 — Neither-Hierarchy / Third-Evidence Feasibility Audit\n\n',
         '完整实验报告｜BCSS validation-only｜C0 seed42 final Epoch25｜zero training\n\n',
         '**结论：第三证据能富集 Both-Wrong，但本轮候选不够可靠，未通过可用性审计。** '
         'Gate A/B/C/D/E/F = FAIL / FAIL / PASS / PASS / PASS / FAIL。'
         'Gate A 失败来自 CandidatePrecision，而非救回数量不足；这是固定决策标签的含义边界，不能误读为“没有第三证据信号”。\n\n']
    body=(f'纯 A0：`{rt["a0"]}`。从该提交新建 `feature/rddr-phase2b111-third-evidence`，PR base=`baseline/official-a0`。'
          '原 `network/`、`tool/`、`train_sshr.py` 不变；只新增审计代码，不引入创新网络。\n\n'
          f'主运行与独立核验 commit：`{rt["code_commit"]}` / `{v["code_commit"]}`。'
          f'冻结合同 SHA256：`{rt["contract_sha256"]}`。\n\n'
          '全部3,418张既有 validation 缓存，native28位置；以下不是重新测得的224×224官方分割指标。\n\n')
    for k,p in rt['paths'].items():body+=f'- `{k}`：`{p}`\n  SHA256：`{rt["source_sha256"][k]}`\n'
    body+=('\n实际执行命令（历史记录；复跑必须更换 output，不覆盖 formal_r1）：\n\n```bash\n'
           'cd /home/duyanhong/DZWdeRepo-rddr-phase2b111\n'+rt['command']+'\n'+v['command']+'\n'
           'RDDR_PHASE2B111_RUN=/home/duyanhong/experiments/RDDR_PHASE2B111/formal_r1 '
           '/home/duyanhong/miniconda3/envs/sshr5090/bin/python -m unittest discover -s tests '
           "-p 'test_rddr_phase2b111*.py' -v\n```\n\n")
    body+=table(['工程项目','实测'],[
        ['环境',f'{rp["gpu"]}；PyTorch {rp["torch"]}；NumPy {rt["numpy"]}'],
        ['主审计耗时',num(rt['total_seconds'])+' s'],['概率-only GPU重放',num(rp['seconds'])+' s（不是网络forward）'],
        ['GPU peak allocated / reserved',num(rp['allocated_bytes']/2**20)+' / '+num(rp['reserved_bytes']/2**20)+' MiB'],
        ['主流程进程 peak RSS',num(rt['peak_process_rss_kib']/2**20)+' GiB'],
        ['测试','54/54，零skip'],['独立检查',f'{len(v["checks"])}/{len(v["checks"])} PASS'],
        ['model forward / backward / autograd / optimizer step / checkpoint write','0 / 0 / 0 / 0 / 0'],
    ])
    body+='本轮新测文件SHA与原始源码一致性；旧模型state/BN/fixed160预测identity原样继承，**不是本轮重测**。所有张量有限；空人口或单标签统计记NA/null，不能用0冒充。'
    out.append(sec(1,'Provenance / SHA / commands',body))
    out.append(sec(2,'Frozen Phase2B1.10 evidence',
        table(['冻结证据','数值'],[['Raw-Wrong / rejected Raw-Wrong','708,407 / 435,185'],['Rejected DW / BW / SW','113,204 / 321,981 / 144,662'],
            ['原 Gate-E 缺口 / residual beneficial','31,266 / 177,865'],['S_D utility / winner / interior image AUC','0.5002 / 0.6083 / 0.5285'],
            ['BW context救回 / 错误第三类','108,541 / 19,165'],['BW内原候选precision','84.9929%']])+
        '上轮决定 `RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED` 保留；S_D recovery 门槛失败不能靠本轮重新阈值化挽回。'
        '上轮84.99%来自 GT-defined Both-Wrong 子群，不是 GT-blind 候选的 precision。'))
    out.append(sec(3,'Scientific roles',
        '`q=JS(ps,pd)/ln2` 是 need，不是 direction；`Delta_sym` 是 shallow/deep 相对裁决；本轮只测试独立问题：'
        '两层级都不足时，邻域是否提出可信替代类别。Phase110中q的utility AUC=0.9067、winner AUC=0.5036，'
        '不能因为q排名某项好就替换本轮主分数。上述是信号的科学角色，不是本轮把创新模块装入A0。'))
    out.append(sec(4,'Residual universe',
        '`U_R = {m_D=0}` 首先在全部3418×784=2,679,712位置上生成，完全不读GT。'
        'all-native U_R有1,925,950位置；事后GT0–3评价人口有1,782,105位置。'
        'GT背景4与ignore255只在评价时排除，不进入候选函数、不作为部署可用mask。'
        '所有科学precision都是前景评价人口口径；背景候选数量仍必须披露，不能声称全像素部署已被验证。'))
    out.append(sec(5,'ctx replay',
        '保持15×15有效图像内邻域、exclude self；ctx_S/ctx_D各自求概率均值，再ctx_sym=0.5(ctx_S+ctx_D)。\n\n'+
        table(['检查','最大绝对误差'],[['四个support与三个context FP32重放','0'],['raw_logits→ps FP32重放','0'],
            ['q重算（原缓存q保留）',f'{rp["errors"]["q"]:.9e} ≤1e-7'],
            ['独立FP64 separable box-filter',f'{v["errors"]["FP64_context"]:.9e} <1e-6']])+
        'FP64交叉检查的舍入差不用于替换冻结context或改变candidate。没有窗口搜索或新教师训练。'))
    counts=rows('candidate_counts')
    out.append(sec(6,'c_s / c_d / c_c',
        '`cs=argmax(ps), cd=argmax(pd), cc=argmax(ctx_sym)`，均沿用first-index tie规则。\n\n'+
        table(['人口','raw argmax ties','deep argmax ties','ctx argmax ties'],[[r['population'],r['raw_argmax_tie'],r['deep_argmax_tie'],r['ctx_argmax_tie']] for r in counts])+
        '这些ties是人口内的argmax并列位置计数；不把它们误称为strict-margin拒绝数。'))
    out.append(sec(7,'Alternative candidate',
        '```text\na_alt = (cc != cs) AND (cc != cd)\n'
        'M_alt = ctx(cc) - max(ctx(cs), ctx(cd))\n'
        'A_alt = (m_D == 0) AND a_alt AND (M_alt > 0)\n```\n\n'
        'A_alt函数输入只有ps/pd/ctx/mD，不含GT。M_alt==0严格拒绝；本次strict-zero拒绝数为0，'
        '与旧“不同于both”提议计数没有tie导致的差异。这里只计算附件规定的诊断候选，未设计额外恢复gate或写回推理。'))
    out.append(sec(8,'M_alt',
        '主分数固定为替代类别相对两个层级候选的context优势。Rescue/gradient任务在A_alt上用原M_alt；'
        'Both-Wrong检测在U_R上用`where(A_alt,M_alt,0)`。不加温度、不乘q、不翻符号、不做排名阈值或Top-k。'
        'Controls在U_R保留各自原始值，不额外乘候选mask。'))
    out.append(sec(9,'Candidate counts',
        table(['人口','全部位置','U_R','候选','候选/U_R','Strict-zero rejected'],
              [[r['population'],cnt(r['total']),cnt(r['universe']),cnt(r['candidate']),pct(r['candidate_rate_in_universe']),r['strict_zero_rejected']] for r in counts])+
        '全部native候选227,781，其中前景202,678、背景25,103、ignore 0。前景候选分布于3,167张图像；'
        '所有3,418张均处理，没有因无前景或无候选而丢图。'))
    out.append(sec(10,'Candidate composition',
        table(['前景候选来源','Count','占全部前景候选'],[[r['group'],cnt(r['count']),pct(r['fraction'])] for r in rows('candidate_composition')])+
        '五个非空组互斥且穷尽，other=0。Stable-Correct误入56,477，占所有候选27.8654%，'
        '是不能只报告Both-Wrong precision的直接证据。'))
    out.append(sec(11,'Deployment-style CandidatePrecision',
        f'`108541 / 202678 = {pct(h["CandidatePrecision"])}`，95% CI **{ci("CandidatePrecision",True)}**。\n\n'
        '这项precision是先GT-blind生成候选、再仅在GT0–3评价，不是先用GT筛选Both-Wrong。'
        '正类108,541、失败94,137；点值<65%，CI下界也未超过55%，Gate A的precision两项均失败。'
        '上轮BW内84.9929%仍可重现，但不能替代本项53.5534%。'))
    hard=rows('hard_effect')
    out.append(sec(12,'Hard Repair / Harm / WrongToWrong',
        '只做内存诊断：候选处raw→cc，其余位置保留raw；不生成新模型预测文件、不改变官方inference。\n\n'+
        table(['评价人口','Repair','Harm','Wrong→Wrong activated','StableCorrect activated','NetRepair','Accuracy delta'],
              [[r['population'],cnt(r['Repair']),cnt(r['Harm']),cnt(r['WrongToWrong_activated']),r['StableCorrect_activated'],cnt(r['NetRepair']),num(float(r['hard_accuracy_delta'])*100)+' pp'] for r in hard])+
        table(['评价人口','Raw accuracy','诊断后accuracy','Wrong→Wrong full','StableCorrect full'],
              [[r['population'],pct(r['raw_accuracy']),pct(r['diagnostic_accuracy']),cnt(r['WrongToWrong_full']),cnt(r['StableCorrect_full'])] for r in hard])+
        f'U_R accuracy delta 95% CI={ci("Hard_accuracy_delta_UR",True)}；全部前景 delta CI={ci("Hard_accuracy_delta_foreground",True)}（对应百分点）。'
        'Hard net=43,535，count-equivalent CI='+ci('Hard_NetRepair_count_equivalent')+'。'
        '净修复为正不能抵消预注册precision/排名失败；这是native28 hard accuracy，不是mIoU提升。'))
    out.append(sec(13,'Coverage versus 31,266 gap',
        table(['项目','结果'],[['ThirdRescueCount',cnt(h['ThirdRescueCount'])],['count-equivalent 95% CI',ci('ThirdRescue_count_equivalent')],
            ['RequiredGap',cnt(h['RequiredGap'])],['Rescue / gap',num(h['rescue_to_gap'])],['该比值95% CI',ci('ThirdRescue_to_gap')]])+
        '数量与下界都足够。**Gate A失败不是数量不足，而是其附带的CandidatePrecision不合格。**'
        '此处hard-label headroom与旧局部导数BenefitRate不是同一事件；不宣称旧Gate E被补齐、或Full25一定获益。'))
    def ranking_table(task):
        r=s['primary'][task]
        return table(['指标','结果'],[['Image-balanced AUROC',num(r['image_auroc'])],['95% CI',ci(task+':all:M_alt:image_AUROC')],
            ['Pooled AUROC',num(r['auroc'])],['AUPRC (noninterpolated AP)',num(r['auprc'])],['Positive prevalence',pct(r['prevalence'])],
            ['正/负',cnt(r['positive'])+' / '+cnt(r['negative'])],['Dual-label eligible / 有targets图像',str(r['eligible_images'])+' / '+str(r['images_with_targets'])],['zero excluded',r['zero_excluded']]])
    out.append(sec(14,'M_alt rescue ranking',ranking_table('rescue')+
        'Task B在全部前景候选内区分Third-Rescue vs Alternative-Failure。Image AUC=0.6249，虽下界>0.50，'
        '但未达0.65，且95%CI上界约0.6349。不能改用pooled或其它score过门槛。'))
    out.append(sec(15,'Both-Wrong detection',ranking_table('bothwrong')+
        f'`P(BW|A_alt)={pct(h["BW_prevalence_candidate"])}`，高于`P(BW|U_R)={pct(h["BW_prevalence_UR"])}`。'
        'Gate C PASS。这个结果支持候选富集neither-hierarchy位置，但不能证明context给出的那个替代类别正确；'
        '检测Both-Wrong与安全选择其正确第三类是不同任务。'))
    body='C_ctx=max(ctx)，E_ctx=-H(ctx)=sum ctx log(ctx+1e-8)，q、Delta沿用冻结值，D_hier=1-max(max ps,max pd)。所有方向固定。\n\n'
    for task in ('rescue','bothwrong','gradient'):
        body+=f'### {task}\n\n'+table(['Score','Pooled AUC','Image AUC','95% CI','AP'],
            [[r['score'],num(r['auroc']),num(r['image_auroc']),ci(task+':all:'+r['score']+':image_AUROC'),num(r['auprc'])]
             for r in rows('score_controls') if r['task']==task])
    body+='没有仅展示最佳对照，没有翻转低AUC的score，也没有拼接置信度、q与M_alt训练分类器。'+source('score_controls')
    out.append(sec(16,'Controls',body))
    out.append(sec(17,'Context KL logit gradient',
        '```text\nt = stopgrad(ctx_sym); p = softmax(L_raw); eps = 1e-8\n'
        'KL = sum_c t_c * [log(t_c+eps) - log(p_c+eps)]\n'
        'r_c = t_c*p_c/(p_c+eps)\n'
        'g_c = p_c*sum_j(r_j) - r_c\nv = -g\n```\n\n'
        '逐候选、未加权、未归一化，不乘q、不聚合训练loss。使用冻结FP32 p/t升至FP64计算解析公式；'
        '非候选g=0。不是简化的p−t，亦没有调用autograd/backward。GT margin按原始logits的全部max-tied nonGT competitor计算精确方向导数。\n\n'+
        table(['独立检查','误差/结果'],[['显式softmax Jacobian vs闭式公式',f'{v["errors"]["jacobian_vs_formula"]:.9e}'],
            ['固定128候选×4通道FP64 logit有限差分',f'{v["errors"]["real128_finite_difference"]:.9e} <1e-6'],
            ['实测candidate dM符号一致性','完全一致，无epsilon重标'],['非候选梯度/dM','精确0']])+
        '真实有限差分从原始logits重算FP64 softmax，只用于验证；没有替换主流程的冻结概率或标签。'
        '本轮每像素未加权导数尺度不能与旧q加权/全图归一化loss的dM幅度直接比较。'))
    def utab(rr):
        return table(['Group','N','Benefit%','Harm%','Zero%','Mean dM','Median dM'],
            [[r['group'],cnt(r['count']),pct(r['benefit_rate']),pct(r['harm_rate']),pct(r['zero_rate']),num(r['mean_dm'],6),num(r['median_dm'],6)] for r in rr])
    out.append(sec(18,'Candidate gradient utility',utab(rows('context_gradient'))+
        '以上所有分组均在candidate内评价，分母不是原整组。全候选有益138,634、有害64,044、zero0。'
        'ThirdRescue全部局部有益；但AlternativeFailure内仍有30,093个局部有益位置，因此hard错误与局部margin下降不是等价标签。'
        'Deep-Win intrusion中soft dM大多有益，同时hard替代类别仍全部错误，不能混用两种“安全”概念。'))
    out.append(sec(19,'M_alt gradient-utility ranking',ranking_table('gradient')+
        f'全候选Benefit={pct(gu["benefit_rate"])} > Harm={pct(gu["harm_rate"])}，Mean dM={num(gu["mean_dm"],6)}>0，'
        'M_alt gradient image AUC=0.6270≥0.60，Gate D PASS。局部方向有效不保证训练稳定、有限步获益或硬预测安全。'))
    def protection(file):
        rr=rows(file)
        return table(['Denominator','N','Active','Activation%','Hard harm%','Third proposal%','Rescue%','Candidate precision'],
            [[r['denominator'],cnt(r['population_count']),cnt(r['candidate']),pct(r['activation_rate']),pct(r['hard_harm_rate']),pct(r['third_intrusion_rate']),pct(r['rescue_rate']),pct(r['candidate_precision'])] for r in rr])+table(
            ['Denominator','Context accuracy','dM Benefit%','dM Harm%','dM Zero%','Mean dM','Median dM'],
            [[r['denominator'],pct(r['unconditional_ctx_accuracy']),pct(r['benefit_rate']),pct(r['harm_rate']),pct(r['zero_rate']),num(r['mean_dm'],6),num(r['median_dm'],6)] for r in rr])+source(file)
    out.append(sec(20,'Raw-Correct protection',protection('rawcorrect_protection')+
        'Gate E使用U_R内全部1,346,920个Raw-Correct作分母，未激活位置dM=0。Hard harm=4.8263%≤8%，'
        'gradient harm=4.6715%≤15%；对应CI分别'+ci('RawCorrect_hard_HarmRate',True)+' / '+ci('RawCorrect_gradient_HarmRate',True)+'。'
        '**但是，被激活的65,006个Raw-Correct位置hard harm=100%，gradient harm=96.7941%。**'
        '这是少量激活下的全分母通过，不是候选对正确语义无害。'))
    out.append(sec(21,'Rejected Deep-Win protection',protection('deepwin_protection')+
        '全分母第三类误入8.8036%，与Phase110一致且≤12%。9,966次候选激活均未硬救回；'
        'raw本来错误，所以raw-correct→wrong定义的hard harm为0，但不能据此说错误第三类安全。'
        'Context accuracy 64.8758%是对整个DW人口不加候选限制的诊断；实际候选context accuracy为0。'))
    out.append(sec(22,'Rejected Shallow-Win protection',protection('shallowwin_protection')+
        '全分母第三类误入5.8958%，与Phase110一致且≤10%。激活的8,529个位置全部损害raw hard预测；'
        'active-only gradient harm=94.5011%。未使用GT防止这些位置被激活。'))
    out.append(sec(23,'Rejected Both-Wrong rescue',protection('bothwrong_rescue')+
        table(['Group','旧提议','旧救回','旧错误','严格候选','严格救回','零margin拒绝'],
            [[r['group'],cnt(r['old_alternative']),cnt(r['old_rescue']),cnt(r['old_wrong']),cnt(r['strict_alternative']),cnt(r['strict_rescue']),r['rejected_zero_margin']] for r in rows('legacy_replay')])+
        '108,541正确第三类与19,165错误第三类均精确重现。这里的84.9929%仍是BW条件precision，'
        '不得用它代替第11节包含one-correct及stable-correct的全候选precision。'))
    out.append(sec(24,'Stable-Correct intrusion',protection('stablecorrect_intrusion')+
        '56,477个双方原本正确的位置被context第三类覆盖，全部hard有害；占全部65,006次hard harm的 '
        +pct(56477/65006)+'。本轮没有使用“双方一致时禁用”等事后规则。'))
    def strata_tables(file):
        rr=rows(file)
        return table(['Group','U_R','Candidate','Activation%','Rescue / failure','Precision','Rescue image AUC [CI]'],
            [[r['group'],cnt(r['universe']),cnt(r['candidate']),pct(r['candidate_rate']),cnt(r['rescue'])+' / '+cnt(r['failure']),pct(r['precision']),
              num(r['image_auroc'])+' '+ci('rescue:'+r['group']+':M_alt:image_AUROC')] for r in rr])+table(
            ['Group','Dual images / rescue power','Gradient AUC / power','dM Benefit% / Harm%','Raw-Correct hard / dM harm%'],
            [[r['group'],r['eligible_images']+' / '+r['power'],num(r['gradient_image_auroc'])+' / '+r['gradient_power'],
              pct(r['benefit_rate'])+' / '+pct(r['harm_rate']),pct(r['rawcorrect_hard_harm'])+' / '+pct(r['rawcorrect_gradient_harm'])] for r in rr])
    out.append(sec(25,'q strata',
        '冻结Q边界：`'+', '.join(str(x) for x in s['q_edges'])+'`，side=left；Top20沿用缓存，不在U_R或candidate中重新计算。\n\n'+
        strata_tables('q_strata')+'Top20 precision约53.36%，并未解决可靠性问题；不实施Top20-only或q阈值规则。'))
    out.append(sec(26,'Per-class',strata_tables('per_class')+
        '四类rescue与gradient排名均POWERED（≥500正、≥500负、≥30dual图像）。只有class0/1的rescue image AUC>0.55，'
        'class2/3未达标。Class2 Raw-Correct hard harm=14.7652%，class3=8.1355%；'
        '虽然全局Gate E通过，不代表每一类都满足全局阈值。这里只披露，不改class规则、不增设事后门槛。'))
    out.append(sec(27,'Boundary / interior',strata_tables('boundary_interior')+
        '复用≤7px boundary，未改窗口/边界宽度。Interior rescue image AUC=0.6480通过F的interior子项；'
        '但boundary precision仅38.7176%，其Raw-Correct hard harm=17.1503%，反映全局平均之外的风险。'
        '不得用interior-only取代global gates。'))
    body=('10,000次paired image bootstrap，seed42，每次3418张有放回抽样，所有endpoint同一批索引。'
          'Image AUC只平均dual-label图像；比例每次重新汇总分子/分母，使用2.5/97.5 percentile CI。\n\n'
          '救回count-equivalent=`draw救回数/draw全部Raw-Wrong数 × 固定708407`，与固定31266比较。'
          'Hard NetRepair count-equivalent=`draw净修复/draw前景U_R × 固定1782105`；另给出U_R与全部前景归一化accuracy delta。\n\n'
          f'索引 SHA256：`{s["bootstrap_rng_sha256"]}`。独立direct-gather相对主流程bincount聚合的replicate误差 '
          f'`{v["errors"]["bootstrap_replicates"]:.3e}`。所有endpoint有效重采样数均为10000。\n\n')
    assert all(int(r['valid_resamples'])==10000 for r in cirows)
    body+=table(['Endpoint（率0–1；count-equivalent为位置数）','Estimate','95% CI'],
        [[r['metric'],num(r['estimate'],6),'['+num(r['ci_low'],6)+', '+num(r['ci_high'],6)+']'] for r in cirows])
    body+='区间仅表示这个固定checkpoint下的图像抽样不确定性，不是多seed方差；不覆盖长期反复方案选择的适应性偏差。未做多重比较校正，controls仅为诊断。'
    out.append(sec(28,'10k bootstrap',body))
    out.append(sec(29,'Gate A–F',table(['Gate','冻结要求','实测','结论'],[
        ['A','救回≥31266且CI下界≥31266；precision≥.65且lower>.55','救回108541/lower104859.94通过；precision.5355/lower.5228失败','FAIL'],
        ['B','rescue image AUC≥.65，lower>.50',num(a['image_auroc'])+' '+ci('rescue:all:M_alt:image_AUROC'),'FAIL'],
        ['C','BW image AUC≥.65，lower>.50，且BW富集',num(b['image_auroc'])+'；63.0093%>18.0675%','PASS'],
        ['D','candidate Benefit>Harm，Mean dM>0，gradient AUC≥.60','68.4011%>31.5989%；mean .240582；AUC .6270','PASS'],
        ['E','RC hard≤8%，RC dM harm≤15%，DW intrusion≤12%，SW intrusion≤10%','4.8263% / 4.6715% / 8.8036% / 5.8958%','PASS'],
        ['F','interior rescue AUC>.60且≥3个powered class AUC>.55','interior .6480通过；仅2/4类通过（4/4 powered）','FAIL'],
    ])+'按A→B/C→D→E→F优先级输出，A已经失败；B与F的失败仍完整披露。不存在仅因power不足而暂停判定的情况。'))
    out.append(sec(30,'Secondary flags',
        '`CONTEXT_CONFIDENCE_DIAGNOSTIC_STRONGER = FALSE`。同一candidate rescue任务的C_ctx image AUC=0.6221，'
        '略低于M_alt=0.6249；尽管C_ctx的pooled AUC略高，也不能替代预注册image AUC比较。'
        '未根据任何control设计新机制。'))
    out.append(sec(31,'STRONG_THIRD_EVIDENCE_SIGNAL',
        '`FALSE`。A/B/F失败；precision .5355<.75、rescue AUC .6249<.75、gradient AUC .6270<.70。'
        '即使救回数量≥2gap、全局RC hard harm≤5%，也不满足Strong的合取要求。'))
    out.append(sec(32,'Scientific interpretation / delivery',
        '本轮将问题定位为：**可以识别更可能“两层级都错”的位置，但仍不能足够可靠地选择正确替代语义。**'
        '第三证据存在，局部soft方向平均有益，硬净修复也为正；这些事实不能替代候选precision和跨类别可靠性门槛。\n\n'
        '关键污染来自双方原本都正确的Stable-Correct位置，以及Shallow-Win位置。全分母保护通过由较低激活率支撑，'
        'active-only仍有严重伤害；boundary和class2更明显。上述仅为机制定位，不在本轮加一致性排除、边界规则、'
        '类别阈值或新score。\n\n'
        '因此不进入Phase2B1.12 gate设计，不训练，不做test/LUAD或其它seed；等待用户下一份独立方案。'
        '实验失败不等于所有context思路不可能，只说明这套冻结候选、分数和合同未达标。\n\n'
        '交付54项测试、29项独立检查、全部CSV/JSON与bootstrap replicates、输入SHA和命令，'
        '[运行说明](README_rddr_phase2b111.md)、[冻结合同](rddr_phase2b111_contract.md)、'
        '[交付摘要](rddr_phase2b111_delivery_summary.md)。证据位于`audit/results/rddr_phase2b111/`；'
        '大输入及image_statistics NPZ留服务器。没有覆盖旧实验、checkpoint或官方代码。'))
    out.append(sec(33,'Exact final decision',
        '固定决策标签如下。再次强调：**operational headroom不足在这里指Gate A的整体可操作条件不成立，'
        '具体短板是precision，不是救回数量。**完整审计结束，停止，不自动追加实验。'))
    out.append('DECISION = '+s['decision']+'\n')
    text=''.join(out);assert sum(x.startswith('## ') for x in text.splitlines())==33
    return text

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True)
    args=ap.parse_args()
    if args.report.exists() or args.manifest.exists():raise FileExistsError('Use fresh report and manifest paths; no overwrite.')
    text=render(args.run);entries={}
    for p in sorted(args.run.iterdir()):
        if p.name.startswith(P) and p.suffix in ('.json','.csv','.txt') and p.name!=P+'delivery_manifest.json' and p.resolve()!=args.manifest.resolve():
            entries[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.manifest.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    entries[args.report.name]=dict(sha256=sha(args.report),bytes=args.report.stat().st_size)
    with args.manifest.open('x',encoding='utf-8',newline='\n') as f:json.dump(dict(schema=1,files=entries),f,indent=2);f.write('\n')
    print(json.dumps(dict(report=str(args.report),sha256=sha(args.report),files=len(entries))))

if __name__=='__main__':main()
