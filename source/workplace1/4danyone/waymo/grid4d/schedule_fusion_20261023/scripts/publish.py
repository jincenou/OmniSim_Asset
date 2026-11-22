"""Publish fixed comparisons, measured costs, code/replay audit and explicit limits."""
import argparse,asyncio,csv,hashlib,html,json,subprocess,time
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1]
LABELS=['N04','N06','N11']
NAMES={'A':'完整 A（52999）','F0':'原融合对照（32999）','F1':'后期降LR（32999）','F2':'早alpha＋后期降LR（32999）','F138':'后期降LR＋5k（37999）','F238':'早alpha＋后期降LR＋5k（37999）'}


def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


async def decode(paths):
    limit=asyncio.Semaphore(3)
    async def one(path):
        async with limit:
            cmd=['/mnt/ojc/miniconda3/envs/qwen35_vlm/bin/ffprobe','-v','error','-threads','1','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,width,height','-of','json',str(path)]
            p=await asyncio.to_thread(subprocess.run,cmd,capture_output=True,timeout=55)
            assert p.returncode==0 and not p.stderr.strip(),p.stderr.decode()
            row=json.loads(p.stdout)['streams'][0];wanted=121 if 'N11' in str(path) else 61
            assert int(row['nb_read_frames'])==wanted
            return {'path':str(path.relative_to(STUDY)),**row}
    return await asyncio.gather(*(one(p) for p in paths))


def main():
    assert (STUDY/'NUMERICAL_COMPLETE.json').exists()
    assert (STUDY/'VISUAL_REVIEW.json').exists(),'Record actual visual scope before publishing'
    final=read(STUDY/'MEANS_FINAL.json')['rows'];gates=read(STUDY/'GATES_FINAL.json')['checks']
    extension=read(STUDY/'DECISION_33K.json')['selected_for_extension'];suffix='_extension' if extension else ''
    media=[STUDY/'comparison'/f'{label}{suffix}'/f'view{view:02d}.mp4' for label in LABELS for view in [0,8,16]]
    decode_rows=asyncio.run(decode(media));states=read(STUDY/'STATUS.json')['jobs'];costs=[]
    methods=[r['method'] for r in final]
    for label in LABELS:
        prefix=read(STUDY/'runs'/f'{label}_prefix/complete.json');middle=read(STUDY/'runs'/f'{label}_middle/complete.json')
        for name in methods:
            if name=='A':continue
            extension_name=name.endswith('38');base=name[:-2] if extension_name else name
            chain=[label+'_prefix']+([] if base=='F2' else [label+'_middle'])+[label+'_'+base]
            seconds=sum(states[k]['finished_at']-states[k]['started_at'] for k in chain)
            if extension_name:seconds+=read(STUDY/'runs'/f'{label}_{name}/complete.json')['process_seconds']
            costs.append({'label':label,'method':name,'actual_updates':37999 if extension_name else 32999,
                'composed_training_process_minutes':seconds/60,'scope':'Measured sum of full required stages including import/load/save; shared prefix counted once per recipe; different GPUs/shared loads; not a controlled speed ratio vs historical A'})
    with (STUDY/'costs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(costs[0]));w.writeheader();w.writerows(costs)
    paired=[]
    for label in LABELS:
        a=read(STUDY/'runs'/f'{label}_F0/restore.json');b=read(STUDY/'runs'/f'{label}_F1/restore.json')
        assert a['parent_sha256']==b['parent_sha256']
        assert read(STUDY/'runs'/f'{label}_F0/sampling.json')==read(STUDY/'runs'/f'{label}_F1/sampling.json')
        p=read(STUDY/'runs'/f'{label}_F0/complete.json');q=read(STUDY/'runs'/f'{label}_F1/complete.json');assert p['points']==q['points']
        paired.append({'label':label,'parent28k_sha256':a['parent_sha256'],'same_camera_sequence':True,'same_fixed_point_count':p['points']})
    for filename in ['CODE_FREEZE.json','EVALUATION_FREEZE.json','EXTENSION_FREEZE.json','POSTPROCESS_FREEZE.json']:
        for path,wanted in read(STUDY/filename)['files'].items():assert hashlib.sha256((STUDY/path).read_bytes()).hexdigest()==wanted,(filename,path)
    assets=[];probes=0;maximum=0.
    for label in LABELS:
        m=read(STUDY/'results'/f'{label}{suffix}/metrics.json')
        for name,p in m['replays'].items():probes+=p['probes'];maximum=max(maximum,p['max_error'])
        for name in methods:
            if name=='A':continue
            run=STUDY/'runs'/f'{label}_{name}';it=38000 if name.endswith('38') else 33000
            assert read(run/'complete.json')['actual_updates']==(37999 if it==38000 else 32999)
            for relative in [f'point_cloud/iteration_{it}/point_cloud.ply',f'deform/iteration_{it}/deform.pth','config.json','asset_camera_data.json','recovery.pt']:
                assert (run/relative).is_file(),relative
            assets.append({'label':label,'method':name,'run':str(run),'iteration':it,'complete_dynamic_asset_requires':'PLY + hash/deform + config/cameras + matching original Grid4D runtime; no arbitrary action or world-scale claim'})
    fresh=sum(read(p)['new_updates'] for p in (STUDY/'runs').glob('*/complete.json'))
    audit={'state':'fixed_experiments_and_evaluation_complete','subjects':LABELS,'new_terminal_assets':len(assets),'fresh_actual_updates':fresh,
        'paired_F0_F1':paired,'replay_probes':probes,'replay_max_error':maximum,'replay_source_image_reads':0,
        'reference_protocol':'Global hashes/crops and all anatomical/temporal denominators match historical A; A globals/parts recomputed within1e-6',
        'videos':decode_rows,'visual_review':'VISUAL_REVIEW.json','code_and_plan_freeze':'passed','extension':extension,
        'scope':'3-identity generated-H development experiment; no claim of universal53k replacement or fair historical-A timing ratio'}
    save(STUDY/'completion_audit.json',audit);save(STUDY/'ASSETS.json',{'assets':assets})
    lines=['Grid4D 融合课程与低学习率细化：实验报告','2026-10-23（北京时间）','',
        '对象：N04/N06/N11。固定原G0/4DAnyone生成输入；只训练后端。主表为三人等权，不是八人均值。',
        'F0：旧融合配方；F1：F0加28k..33k学习率log-linear降至0.25；F2：alpha提前至10k..13k，再做相同后期细化。',
        'F0/F1共用完整28k状态与相机序列，点数相同；F2从同一10k状态分叉，允许alpha影响之后的原生增密。',
        'alpha最大0.1；RGB/SSIM/原时空正则/SH/分辨率/帧数/相机/seed均不改变。',
        '首轮所有终态32999实际更新；有条件追加仅5000次，不做后续扫描。',
        '','方法 | 前景PSNR↑ | 全图SSIM↑ | 裁剪LPIPS↓ | 轮廓外alpha↓']
    for r in final:lines.append(f"{NAMES.get(r['method'],r['method'])} | {r['foreground_psnr']:.5f} | {r['ssim']:.6f} | {r['crop_lpips']:.6f} | {r['outside_alpha_mean']:.6f}")
    lines+=['','逐人保护线（NA不通过；数值通过不是完整视觉/使用验收）：']
    for g in gates:lines.append(f"{g['label']} {NAMES.get(g['method'],g['method'])}: {g['numerical_state']}; failed={','.join(g['failed']) or '无'}; unknown={','.join(g['unknown']) or '无'}")
    lines+=['','耗时：必需阶段的训练子进程累计墙钟分钟，含导入/加载/保存，共享前段已计入各成品；不含前端/排队/评测。']
    for c in costs:lines.append(f"{c['label']} {c['method']}: {c['composed_training_process_minutes']:.2f}分钟")
    lines+=['不同GPU和其他负载不同，不能把此表与历史A时长相除宣称公平加速倍率。减少37.74%更新不等于同百分比端到端提速。',
        '',f"本轮新增真实更新总计{fresh}；新终态资产{len(assets)}个不是新增人物数；独立训练视角回放{probes}观测，最大误差{maximum}。",
        '固定H={0,8,16}的39观测计算全局指标；部位/光流用完整时间轴。H为已反复使用的开发参考，不是真实Waymo新视角真值。',
        '可信核心覆盖不是整条腿或手臂完整率；小腿/脚缺失和颜色污染可能在数值通过时仍存在。',
        '', '结论与实际视觉范围见 VISUAL_REVIEW.json；主线是否替换以数值与已核查质量共同判断，不选逐人best掩盖统一配方失败。',
        '没有重训4DAnyone，没有更换renderer、Adam或新模型；原资产和human Git实现未覆盖。',
        '', '调研：RESEARCH.txt；协议：PLAN.json；逐人指标：metrics_final.csv；数值门槛：GATES_FINAL.json；成本：costs.csv；资产：ASSETS.json。']
    (STUDY/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>Grid4D 融合训练对照</title><style>body{font:16px system-ui,sans-serif;max-width:1500px;margin:25px auto;padding:15px;line-height:1.7}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:7px}video,img{max-width:100%}section{margin:40px 0}</style><h1>三个人体：融合课程与低学习率细化</h1>',
        '<p>完整A历史质量参照；F0新融合对照；F1末段降LR；F2早alpha＋末段降LR。若显示F1+5k，为冻结规则触发的唯一共同延长。不是八人均值，也不是真实Waymo新视角指标。</p>',
        '<p><a href="REPORT.txt">详细报告</a> · <a href="RESEARCH.txt">优化调研</a> · <a href="metrics_final.csv">逐人指标</a> · <a href="GATES_FINAL.json">保护线</a> · <a href="costs.csv">耗时</a> · <a href="ASSETS.json">资产</a> · <a href="completion_audit.json">验收</a></p>',
        '<table><tr><th>方案</th><th>前景PSNR↑</th><th>全图SSIM↑</th><th>裁剪LPIPS↓</th><th>外部alpha↓</th></tr>']
    for r in final:page.append(f"<tr><td>{NAMES.get(r['method'],r['method'])}</td><td>{r['foreground_psnr']:.5f}</td><td>{r['ssim']:.6f}</td><td>{r['crop_lpips']:.6f}</td><td>{r['outside_alpha_mean']:.6f}</td></tr>")
    page.append('</table><p>下面每个视频从左至右：生成参考、完整A、F0、F1、F2，以及存在时的5k延长。已数值通过仍可能继承模糊、缺脚等问题，实际抽样视觉范围见报告。</p>')
    for label in LABELS:
        folder=f'comparison/{label}{suffix}';page.append(f'<section><h2>{label}</h2>')
        for view in [0,8,16]:page.append(f'<p>固定视角 {view:02d}</p><video controls preload="none" src="{folder}/view{view:02d}.mp4"></video>')
        page.append(f'<img loading="lazy" src="{folder}/contact.jpg"></section>')
    (STUDY/'index.html').write_text('\n'.join(page))
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
