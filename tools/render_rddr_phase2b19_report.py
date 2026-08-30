"""Source-backed 41-section Chinese report; stdlib only; refuses overwrites."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
P='rddr_phase2b19_'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--output',required=True);ap.add_argument('--manifest',action='store_true');args=ap.parse_args()
    root=Path(args.results);dest=Path(args.output)
    if dest.exists():raise FileExistsError(dest)
    def js(name):return json.loads((root/(P+name+'.json')).read_text(encoding='utf-8'))
    def rows(name):
        with (root/(P+name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    s=js('summary');rt=js('runtime');v=js('verification');dt=js('detach_audit');ident=js('identity_audit');smoke=js('bf16_smoke');selection=js('selection')
    test=(root/(P+'tests.txt')).read_text();assert v['status']=='PASS' and 'Ran 52 tests' in test and test.rstrip().endswith('OK') and 'skipped' not in test
    pr=s['primary'];boot={r['metric']:r for r in rows('bootstrap')};adj=s['adjudication']
    grad=rows('gradient_controls');rec=rows('brr_hhcr');hist=rows('historical_comparison');soft=rows('soft_directional')
    def pct(x):return f'{100*float(x):.4f}%'
    percent={'benefit_rate','harm_rate','zero_rate','active_gradient_fraction','active_transfer_fraction','ActiveTransferFraction','DeepCaptureRate','ShallowProtectionRate','DeepSelectionPrecision',
        'BRR','HHCR','DBR','DCR','raw_accuracy','deep_accuracy','deep_raw_delta','ADT_rate','RG_rate','rate_difference','RG_benefit','RG_harm','dQ_negative_fraction','dQ_zero_fraction'}
    def fmt(x,key=''):
        if x is None or x=='':return 'NA'
        if isinstance(x,bool):return str(x)
        try:
            f=float(x)
            if not math.isfinite(f):return 'NA'
            if key in percent:return pct(f)
            if f==int(f) and key not in ('mean_dm','mean_dQ','rms','mean_pixel_l2','max_abs'):return str(int(f))
            return f'{f:.10g}'
        except (ValueError,TypeError):return str(x).replace('|','\\|')
    def table(rr,cols):
        rr=list(rr)
        return '\n'.join(['| '+' | '.join(title for title,key in cols)+' |','| '+' | '.join('---' for _ in cols)+' |']+['| '+' | '.join(fmt(r.get(k),k) for _,k in cols)+' |' for r in rr])+'\n\n'
    def ci(key,percent=False):
        r=boot[key];f=pct if percent else lambda x:f'{float(x):.10g}'
        return f"{f(r['estimate'])} [{f(r['ci_low'])}, {f(r['ci_high'])}]"
    sections=[]
    def sec(title,text):sections.append(f'## {len(sections)+1}. {title}\n\n{text.strip()}\n')
    gc=[('组','stratum'),('Loss','loss'),('N','targets'),('Benefit','benefit_rate'),('Harm','harm_rate'),('Zero','zero_rate'),('Mean dM','mean_dm'),('Median dM','median_dm')]
    hc=[('组','stratum'),('Loss','loss'),('BRR','BRR'),('HHCR','HHCR'),('DBR','DBR'),('DCR','DCR')]
    ac=[('组','stratum'),('Deep-Win','Deep_Win_count'),('Shallow-Win','Shallow_Win_count'),('双标签图像','dual_label_images'),('Image AUC','image_auroc'),('DeepCapture','DeepCaptureRate'),('ShallowProtection','ShallowProtectionRate'),('BA','BA')]
    sec('Provenance / SHA256 / commands',f'''本报告对应已确认合同的完整执行，不是训练结果。唯一 checkpoint 为 C0 Full25 BCSS seed42，全部3418张 validation；无 test、LUAD、train split、optimizer 或参数更新。

纯 A0：`{rt['a0']}`。GPU执行 commit：`{rt['code_commit']}`；独立复核 commit：`{v['code_commit']}`。
独立分支：`feature/rddr-phase2b19-directional-transfer`；PR目标：`baseline/official-a0`，不自动合并。
合同文件 SHA256：`{selection['contract_sha256']}`。

'''+table([dict(asset=k,path=rt['paths'][k],sha=h) for k,h in rt['source_sha256'].items()],[('资产','asset'),('服务器路径','path'),('SHA256','sha')])+f'''
新 observation SHA256：`{rt['observation_sha256']}`。
随机门 SHA256：`{rt['random_gate_sha256']}`。

实际命令（已执行，重放时必须使用新输出目录）：

```bash
{rt['command']}
{s['command']}
{v['command']}
```

只在独立 tools/tests/docs/results 中新增审计；原 network/tool/train_sshr.py 与 A0 无差异。旧实验全部保留。''')
    sec('Frozen evidence',f'''复用 Phase2B1 native28、Phase2B1.5 对称裁决、Phase2B1.8 raw logits / q导数 / PRG梯度。
冻结裁决 image-balanced AUROC={adj['image_auroc']:.10f}，DeepCapture={pct(adj['DeepCaptureRate'])}，ShallowProtection={pct(adj['ShallowProtectionRate'])}。
上一轮 PRG 全部 Deep-Win 几乎均有益，但 Shallow-Win HHCR=96.2258%。该结果不是本轮可调条件。
原 Phase2B1.8 raw native28 mIoU=43.6349%、teacher=59.3171% 仅是历史语义诊断，不能与原官方最终融合 mIoU 混为一谈。''')
    sec('Convex teacher failure mechanism','''忽略极小的 epsilon 项，convex teacher 对 raw 的 KL 梯度方向近似为 `wD*(ps-pd)`：context 权重主要缩放 deep-directed 更新，并没有为 shallow 独有正确信息提供显式拒绝通道。本轮不再构造该 teacher，也不增加共享 head 对照。历史 PRG 只读取已冻结梯度用于比较。''')
    sec('Directional transfer hypothesis','''`q` 只表示 Need，`Delta_sym` 判断哪一层得到 context 支持，`mD` 决定是否允许 deep→shallow。
只检验冻结点的局部机制：选择性减少有害迁移，同时覆盖足够多的 raw 错误；不推断25轮训练最终 mIoU。''')
    sec('Tensor / precision / denominator contract',table([dict(name='F28_raw',shape='[B,512,28,28]',role='HFRM28_1之前'),dict(name='L_s / p_s / L_d / p_d',shape='[B,4,28,28]',role='复用ic1/deep head'),dict(name='q / Delta / mD',shape='[B,28,28]',role='全部detach')],[('Tensor','name'),('Shape','shape'),('作用','role')])+'''
真实网络 BF16，概率/loss/logit导数 FP32，统计与内积 FP64。主审计 batch1；loss 分母包括全部784个位置，不借GT排除背景。batch20 smoke 使用整批分母。
诊断分母只含 GT0–3（2,479,143位置），background4/ignore255 排除；未使用官方 background overwrite 修正诊断。拒绝位置计零，不从分母消失。''')
    sec('Delta / raw-logit replay',table([dict(item=k,error=x) for k,x in rt['parity'].items()],[('项目','item'),('最大绝对差','error')])+f'''
`S_S=.5*(T_SS+T_SD)`；`S_D=.5*(T_DS+T_DD)`；`Delta=S_D-S_S`。
存储Delta、完整四项support、现场重算Delta完全一致，gate mismatch={rt['recomputed_gate_mismatches']}。旧 raw logits与当前head/frozen-head完全一致。
q重算的最大差为5.96046448e-8，在预先批准1e-7内；这是继承浮点算序差异，主loss始终使用冻结缓存q，不替换。独立q导数精确相同。''')
    sec('DeepCapture / ShallowProtection',table(rows('adjudication_replay')[:1],ac)+f'''
Image AUROC 95%CI：{ci('Delta_image_AUROC')}；DeepCapture：{ci('DeepCapture',True)}；ShallowProtection：{ci('ShallowProtection',True)}。
Image AUROC按同时存在两类冲突的3180张图像等权平均；不是 pooled AUROC（{adj['pooled_auroc']:.10f}）。捕获/保护率按像素池化，bootstrap仍以图像为重采样单位。''')
    sec('Hard direction gate mD','''`mD = 1[Delta_sym > 0]`，精确零点，tie拒绝。无偏置、温度、置信度筛选、类别规则或q阈值。
“preserve shallow”在这里指该位置没有直接蒸馏梯度；若未来真的更新共享网络参数，并不保证这些位置的预测绝对不变。本轮无更新，状态与预测恒等另行实测。''')
    sec('Primary ADT',r'''`L_ADT=sum(q*mD*KL(pd||softmax(L_student)))/(sum(q*mD)+1e-8)`。
`L_student=conv2d(F28_raw,ic1.weight.detach(),ic1.bias.detach())`。
pd/q/Delta/mD全部detach；只有批准的浅层学生路径接收梯度。KL沿用epsilon放在log内部的旧实现，独立FP64解析式验证通过。''')
    sec('UDT control','''`L_UDT=sum(q*KL(pd||ps))/(sum(q)+eps)`。无方向拒绝；仍然有冻结q加权，因此 UDT不是无q加权的普通KL。系数固定1。''')
    sec('Rate-matched RG control','''固定NumPy default_rng42、冻结图像顺序，一次随机实现。每张图在全部784位置中无放回选取与ADT相同数量的位置；不读取GT、q或类别，不反复抽签。
`L_RG=sum(q*m_rand*KL(pd||ps))/(sum(q*m_rand)+eps)`。严格匹配的是全部位置的逐图数量，前景/冲突子组的比例允许自然不同，必须原样报告。'''+table(rows('random_gate')[:1],[('范围','stratum'),('位置数','targets'),('ADT rate','ADT_rate'),('RG rate','RG_rate'),('差值','rate_difference'),('逐图完全匹配','per_image_exact_parity')]))
    sec('Secondary SDT','''`aD=relu(Delta_sym)`；`L_SDT=sum(q*aD*KL(pd||ps))/(sum(q*aD)+eps)`。
不增加temperature/power/额外normalization。它仅是附件预注册的次要探针，不能替代ADT主判定；不会因为次要结果有利而改变本轮合同。'''+table([r for r in soft if r['stratum'] in ('all','Top20','Deep-Win','Shallow-Win','Raw_Wrong')],gc))
    sec('Transfer coverage',table(rows('transfer_coverage'),[('组','stratum'),('全部分母','targets'),('Selected','selected'),('Rejected','rejected'),('激活率','ActiveTransferFraction')])+f'''
总体前景激活{pct(pr['all']['active_transfer_fraction'])}，Raw-Wrong激活{pct(pr['Raw_Wrong']['active_transfer_fraction'])}。后者低于40%，因此在当前冻结门下，即使每个被选中的raw错误都获得正dM，也达不到Gate E要求的40%全分母BenefitRate。此为事后解释，不用于修改门控。''')
    sec('Selected / rejected semantic quality',table([r for r in rows('selected_region_quality') if r['stratum'] in ('all','Top20','Deep-Win','Shallow-Win','Both-Wrong','Stable-Correct')],[('组','stratum'),('选择','selection'),('N','targets'),('Raw acc','raw_accuracy'),('Deep acc','deep_accuracy'),('差值','deep_raw_delta'),('Repair','repair'),('Harm','harm'),('NetRepair','net_repair')])+'''
Repair/Harm是同一冻结点 raw与deep的hard预测交换计数，不是实际参数更新后的纠错数。完整所有组见selected_region_quality.csv。''')
    sec('DeepSelectionPrecision',table(rows('adjudication_precision'),[('组','stratum'),('Precision','DeepSelectionPrecision'),('Recall','DeepCaptureRate'),('Shallow保护','ShallowProtectionRate')])+f'''
在exactly-one-correct集合：Precision={ci('DeepSelectionPrecision',True)}。分母仅selected Deep-Win+selected Shallow-Win，不能解释为全部选中像素的准确率。''')
    sec('GT-margin exact directional derivative','''`v=-dL/dLs`，不做单位范数归一化；`M=Ls[GT]-max_nonGT(Ls)`。
若最大竞争类别打平，dM使用这些当前并列类别中最大的v作为max方向导数，不随意挑一个argmax。共有2111个前景位置存在竞争logit并列。
Benefit/Harm/Zero分别严格使用dM>0/<0/==0。dM是独立logit坐标的局部方向量，不等同于共享参数实际更新后的mIoU或最终训练收益。''')
    sec('All / Top20 directional utility',table([r for r in grad if r['stratum'] in ('all','Top20') and r['loss']!='PRG_previous'],gc)+f'''
ADT all Mean dM CI：{ci('ADT:all:mean_dm')}；Top20：{ci('ADT:Top20:mean_dm')}。
all、Top20及四类Mean dM均为正（6/6），两组Benefit均高于Harm，Gate B通过。大量Zero是门控设计所致，不得去掉后重新宣称主收益覆盖率更高。''')
    sec('Deep-Win transfer',table([r for r in grad if r['stratum']=='Deep-Win'],gc)+table([r for r in rec if r['stratum']=='all'],hc)+f'''
Deep-Win总数314730；ADT Benefit={pct(pr['Deep-Win']['benefit_rate'])}，Harm=0；被拒绝的35.9686%计Zero。DBR分母仍是全部Deep-Win。''')
    sec('Shallow-Win protection',table([r for r in grad if r['stratum']=='Shallow-Win'],gc)+f'''
Shallow-Win总数182899；拒绝率{pct(adj['ShallowProtectionRate'])}，Harm={pct(pr['Shallow-Win']['harm_rate'])}。
Rate下降不代表每次错误迁移的强度下降：ADT Mean dM={pr['Shallow-Win']['mean_dm']:.10g}，仍为负。''')
    sec('BRR_ADT',f'''`BRR=P(mD=1 AND dM>0 AND dQ<0 | Deep-Win)`，全部Deep-Win为分母。
实测{ci('BRR_ADT',True)}，超过0.60，Gate C通过。它低于历史PRG近100%，即本轮明确牺牲一部分纠错覆盖以保护shallow，而非无代价保留全部Deep-Win收益。''')
    sec('HHCR_ADT',f'''`HHCR=P(mD=1 AND dM<0 AND dQ<0 | Shallow-Win)`，全部Shallow-Win为分母。
实测{ci('HHCR_ADT',True)}，低于0.30，Gate D通过。
`gq=dq/dLs`由独立图求导；主loss的q已detach。所有被拒绝位置的dQ严格为0。''')
    sec('PRG versus ADT historical comparison',table(hist,gc+[("BRR","BRR"),("HHCR","HHCR")])+'''
HHCR从96.2258%降至20.1598%，减少约76.07个百分点；Deep-Win BRR从99.9994%降至64.0311%。
必须同时报告：Shallow-Win Mean dM从-0.0012107108变为-0.0021348160，更负。各loss都有自身权重和分母，ADT选区归一化会改变强度，所以不能仅按rate下降推断所有风险均改善，也不能把Mean dM更大直接换算成mIoU提升。''')
    sec('Raw-Correct safety',table([r for r in grad if r['stratum']=='Raw_Correct'],gc)+f'''
ADT Harm={ci('ADT:Raw_Correct:harm_rate',True)}，满足≤30%。本轮No-Go不是因为raw正确位置的总体受害比例超标。''')
    sec('Raw-Wrong benefit coverage',table([r for r in grad if r['stratum']=='Raw_Wrong'],gc)+f'''
ADT Benefit={ci('ADT:Raw_Wrong:benefit_rate',True)}，预注册要求≥40%，少{40-100*pr['Raw_Wrong']['benefit_rate']:.4f}个百分点，CI上界仍低于40%。
这是Gate E唯一失败子项。Raw-Wrong拒绝/零梯度比例高，主全分母覆盖不足；不得用active-only高Benefit替代。''')
    sec('Both-Wrong',table([r for r in grad if r['stratum']=='Both-Wrong'],gc+[('Mean dQ','mean_dQ')])+'''
此组deep与raw的argmax均错误，deep准确率按定义为0。正dM仅表示GT局部margin可能改善，不是deep已经正确。不引入第三证据或特殊修复分支。''')
    sec('Stable-Correct',table([r for r in grad if r['stratum']=='Stable-Correct'],gc+[('Mean dQ','mean_dQ')])+'''
二者argmax相同且都正确并不保证蒸馏方向对GT margin有利；完整展示harm/zero，不能仅展示selected的优势。''')
    sec('q × direction grid',table(rows('q_direction_grid'),[('Q','quintile'),('mD','gate'),('N','targets'),('Raw acc','raw_accuracy'),('Deep acc','deep_accuracy'),('差值','deep_raw_delta'),('Benefit','benefit_rate'),('Harm','harm_rate'),('Mean dM','mean_dm'),('Mean dQ','mean_dQ')])+'''
分位边界完全复用冻结资产，不选择有利cell训练。mD=0各格的直接dM/dQ严格为零。''')
    sec('Per-class / power',table(rows('per_class'),ac+[('Power','power')])+table(rows('per_class'),[('Class','stratum'),('激活','active_transfer_fraction'),('Benefit','benefit_rate'),('Harm','harm_rate'),('Mean dM','mean_dm'),('BRR','BRR'),('HHCR','HHCR')])+'''
Power同时要求≥500 Deep-Win、≥500 Shallow-Win、≥30双标签图像。class3只有418个Shallow-Win，明确UNDERPOWERED；class3总体Mean dM为正也不能弥补该层级安全性证据不足。
class2/3裁决弱于class0/1，表中如实保留，不创建类别规则。依合同该标记不擅改A–G门槛。''')
    sec('Boundary / interior',table(rows('boundary_interior'),ac)+table(rows('boundary_interior'),[('组','stratum'),('Benefit','benefit_rate'),('Harm','harm_rate'),('BRR','BRR'),('HHCR','HHCR'),('Mean dM','mean_dm')])+'''
沿用冻结的boundary≤7px/interior>7px映射，不重估边界宽度或阈值。''')
    sec('Gradient localization',table(rows('gradient_localization'),[('Loss','loss'),('Top20 meanG','Top20_mean_G'),('Bottom80 meanG','Bottom80_mean_G'),('Ratio','ratio')]+[(f'Q{k}',f'Q{k}_mean_G') for k in range(1,6)])+'''
G为四类logit梯度L2，Top20与全局Q5不是同一选择方式；两套冻结分组分别报告，不互相替换。''')
    sec('Active-only diagnostic',table([r for r in rows('active_only') if r['stratum'] in ('all','Top20','Deep-Win','Shallow-Win','Raw_Correct','Raw_Wrong')],[('组','stratum'),('Loss','loss'),('Active N','targets'),('Benefit','benefit_rate'),('Harm','harm_rate'),('Mean dM','mean_dm')])+'''
本表仅说明被选中迁移的质量，不能覆盖主Gate使用的all-denominator。SDT与ADT有相同正权重支撑集，主要改变幅度，不能靠此表解除Raw-Wrong覆盖不足。''')
    sec('ADT versus random gating',table([r for r in rows('random_gate') if r['stratum'] in ('all','Top20','Deep-Win','Shallow-Win','Raw_Wrong')],[('组','stratum'),('ADT rate','ADT_rate'),('RG rate','RG_rate'),('RG Mean dM','RG_mean_dm'),('RG Benefit','RG_benefit'),('RG Harm','RG_harm')])+f'''
预注册三项配对差（ADT−RG）：

- all Mean dM：{ci('ADT-RG:all:mean_dm')}。
- Shallow-Win Harm：{ci('ADT-RG:Shallow-Win:harm_rate',True)}（负值有利）。
- Deep-Win Benefit：{ci('ADT-RG:Deep-Win:benefit_rate',True)}。

三项95%CI均支持ADT，且满足点估计条件，Gate F通过。采用事先固定的OR规则，不声称做过多重比较家族错误率校正；随机对照只有一个seed42 realization，不等同跨随机门seed稳定性。''')
    sec('Feature / upstream parameter gradients',table(rows('feature_gradient'),[('组','stratum'),('N','targets'),('RMS','rms'),('Mean pixel L2','mean_pixel_l2'),('Maxabs','max_abs'),('Finite','finite')])+table(rows('parameter_gradient'),[('参数','parameter'),('Numel','numel'),('Energy','total_squared_energy'),('RMS','rms'),('Maxabs','max_abs'),('非零图像','nonzero_images')])+'''
39个批准参数包含b4..b4_5及bn45；各卷积子组均有非零梯度，ic1两项始终零。BN始终eval，仅诊断时允许其affine求导；这不是修改未来原始训练freeze规则。Feature能量与parameter能量不混作同一分母。''')
    sec('Detach / no-GT / no-step audit',table([dict(check=k,value=dt[k]) for k in ('q_detached','delta_detached','gate_detached','deep_source_detached','primary_ic1_none','hfrm_none','upstream_conv_nonzero','all_other_primary_gradients_none','rejected_feature_zero','rejected_logit_zero','optimizer_created','optimizer_steps','checkpoint_written')],[('检查','check'),('值','value')])+'''
主loss无GT输入，q导数只在独立诊断图计算。深层共享祖先收到合法浅层梯度不被误判为deep-target漏梯度；deep输出、HFRM和head均不接收本轮loss梯度。''')
    sec('Batch20 BF16 / runtime / memory',table([dict(item=k,value=smoke[k]) for k in ('batch','loss','active_transfer_fraction','seconds','allocated_bytes','reserved_bytes','upstream_conv_energy','head_energy','all_finite','pass')],[('项目','item'),('值','value')])+f'''
Batch20 peak allocated={smoke['allocated_bytes']/1024**3:.4f}GiB，reserved={smoke['reserved_bytes']/1024**3:.4f}GiB，低于22GiB预设预算。
GPU全量流程{rt['total_seconds']:.3f}s，其中冻结重放{rt['parity_seconds']:.3f}s、3418张真实ADT回传{rt['gradient_seconds']:.3f}s；统计/bootstrap {s['analysis_seconds']:.3f}s。
环境：{rt['gpu']}，PyTorch{rt['torch']}，NumPy{rt['numpy']}。这些是审计耗时，不是25轮训练速度预测。Batch20 smoke是选定b4-path的内存，不代表未来全网络训练显存。''')
    sec('Zero-update identity',table([dict(item=k,value=ident[k]) for k in ('state_before','state_after','bn_before','bn_after','checkpoint_sha_before','checkpoint_sha_after','state_unchanged','bn_unchanged','prediction_unchanged','raw_fixed160_exact')],[('项目','item'),('值','value')])+f'''
固定32+seed42随机128，共160张重放；官方推理前后hash相同：`{ident['prediction_before']['sha256']}`，{ident['prediction_before']['pixels']}像素。hash在官方background overwrite之前取得。
Checkpoint strict load missing/unexpected={ident['missing_keys']}/{ident['unexpected_keys']}。全部3418 gradient finite；权重/BN未更新。''')
    sec('Bootstrap / independent verification / tests',table(rows('bootstrap'),[('指标','metric'),('Estimate','estimate'),('CI low','ci_low'),('CI high','ci_high'),('有效次数','valid_resamples')])+f'''
固定10000 image-level paired resamples，seed42。指标按各自估计量重算：AUC对双标签图像均值，其余池化分母；不做pixel bootstrap。RNG序列SHA=`{s['bootstrap_rng_sha256']}`。
独立验证不导入主审计/分析模块：另写FP32 loss/梯度重放、FP64 epsilon-KL/JS解析式、梯形AUROC和直接gather bootstrap。

'''+table([dict(check=k,pass_=x) for k,x in v['checks'].items()],[('独立检查','check'),('PASS','pass_')])+table([dict(item=k,error=x) for k,x in v['errors'].items()],[('误差','item'),('最大绝对差','error')])+'''
52项测试全部PASS、0 skips，覆盖附件规定33项测试，并增加公式、全拒绝和判定优先级检查。完整日志见rddr_phase2b19_tests.txt。''')
    gates=[dict(gate='A',rule='AUC≥.75 / capture≥.60 / protection≥.75 / BA≥.70',actual=f"{adj['image_auroc']:.6f} / {adj['DeepCaptureRate']:.6f} / {adj['ShallowProtectionRate']:.6f} / {adj['BA']:.6f}"),
        dict(gate='B',rule='all与Top20 Benefit>Harm；≥5/6 Mean>0',actual='两组通过；6/6'),
        dict(gate='C',rule='BRR≥.60 / DW Benefit≥.60 / Mean>0',actual=f"{s['BRR']:.6f} / {pr['Deep-Win']['benefit_rate']:.6f} / {pr['Deep-Win']['mean_dm']:.8g}"),
        dict(gate='D',rule='HHCR≤.30 / SW Harm≤.30 / protection≥.70',actual=f"{s['HHCR']:.6f} / {pr['Shallow-Win']['harm_rate']:.6f} / {adj['ShallowProtectionRate']:.6f}"),
        dict(gate='E',rule='RawCorrect Harm≤.30 / RawWrong Benefit≥.40 / active≥.10',actual=f"{pr['Raw_Correct']['harm_rate']:.6f} / {pr['Raw_Wrong']['benefit_rate']:.6f} / {pr['all']['active_transfer_fraction']:.6f}"),
        dict(gate='F',rule='Mean优于RG + 至少一冲突rate改善 + 至少一配对CI有利',actual='三个CI均有利；点条件均满足'),dict(gate='G',rule='finite/detach/批准梯度/BF16/identity/no optimizer',actual='全部通过')]
    for r in gates:r['status']=s['gate_'+r['gate']]
    sec('Gate A–G',table(gates,[('Gate','gate'),('冻结要求','rule'),('观测','actual'),('结果','status')])+'''
Gate E仅Raw-Wrong BenefitRate失败，其余六个Gate均通过。按已确认优先级，G→A→任一B/C/D/E→F→GO；不得因为其他Gate多数通过而改变判定。''')
    sec('Secondary / strong flags',f'''SOFT_DIRECTIONAL_TRANSFER_PROMISING={s['SOFT_DIRECTIONAL_TRANSFER_PROMISING']}。
STRONG_DIRECTIONAL_TRANSFER_SIGNAL={s['STRONG_DIRECTIONAL_TRANSFER_SIGNAL']}。

'''+table([r for r in grad if r['loss'] in ('ADT','SDT') and r['stratum'] in ('all','Shallow-Win','Raw_Wrong')],gc)+'''
SDT满足预先给定的三项次要标准，但本轮仍以ADT A–G判定；Raw-Wrong主覆盖不足不能由soft flag推翻。Strong条件要求Raw-Wrong Benefit≥50%，本轮不满足。''')
    sec('Scientific interpretation / limitations','''本轮证明了什么：对称裁决可以作为有信息的方向开关；相较随机同数量选择，ADT在预注册三项差值上均获得有利CI；大量Shallow-Win位置被拒绝，局部有害迁移发生率显著减少。

本轮没有证明什么：并未形成通过全部readiness门槛的机制。Raw-Wrong纠错覆盖只有35.5865%，即便被选中部分质量较好，仍未达到40%；Shallow-Win剩余更新的平均负向margin幅度更大；class3层级安全性UNDERPOWERED。

因此“裁决有用”不等于“现在可以训练”。方向选择和覆盖之间的权衡仍存在；本轮不会调整阈值、放开某类别、增加第三证据、挑Top20训练、改变loss尺度或改随机seed。局部logit梯度不替代实际优化过程及最终分割评价。所有结论限于C0 seed42固定checkpoint和这套validation诊断。''')
    sec('Exact decision / stop',f'''A/B/C/D/E/F/G = {' / '.join(s['gate_'+k] for k in 'ABCDEFG')}。

判定：`{s['decision']}`。

停止在完整审计报告，未训练、未选择lambda、未运行test/LUAD、未创建新checkpoint。后续若改变合同，需要独立提出并审核；本轮不自动补救失败项。

机器可读输出与CSV位于`audit/results/rddr_phase2b19/`；服务器大缓存位于`{s['run']}`，无多GB特征张量落盘。''')
    assert len(sections)==41
    text='# RDDR-Net Phase-2B1.9 Directional Transfer Audit\n\n日期：2026-08-30。Validation-only / zero-training / zero-optimizer-step。\n\n'+ '\n'.join(sections)+'\nDECISION = '+s['decision']+'\n'
    assert len(re.findall(r'^## \d+\.',text,re.M))==41
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text,encoding='utf-8',newline='\n')
    if args.manifest:
        target=root/'artifact_manifest.json'
        if target.exists():raise FileExistsError(target)
        payload=dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.iterdir()) if p.is_file()},report=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest)))
        target.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(dest),sections=41,sha256=sha(dest),decision=s['decision']),ensure_ascii=False))

if __name__=='__main__':main()
