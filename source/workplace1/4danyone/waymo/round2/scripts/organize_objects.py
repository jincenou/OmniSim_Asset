"""Build a read-only browsing layer without moving or editing source artifacts.

Dry-run is the default. Publishing is explicit; rollback only removes entries
listed in this tool's publication inventory and never follows symbolic links.
"""
import argparse
import csv
import fcntl
import hashlib
import html
import io
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from backend_variants import completed_variants

R=Path(__file__).resolve().parents[1];W=R.parent;O=W/'objects'
SHORT={'r07_short61_24','m02_short61_24'}
STAGES=('01_input','02_preprocess','03_motion','04_generation','05_reconstruction',
        '06_assets','07_demo','08_evaluation','09_records')
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,text):
    tmp=p.with_name(p.name+'.publishing');tmp.write_text(text);tmp.replace(p)
def jsontext(v):return json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
def record(label):
    walking=label=='walk063_rootlocal24'
    if walking:
        source=R/'experiments/walk063_rootlocal24_v1/outputs/turbo'
        original=W/'experiments/pilot_063';asset=R/'reconstruction/outputs'/label
        demo=asset/'demo';mapping=original/'inputs/frame_mapping.json'
        job={'source':str(source),'input':str(original/'inputs/main.mp4'),'mapping':str(mapping),
             'native_mapping':str(original/'inputs_native/frame_mapping.json'),
             'data':str(R/'reconstruction/data'/label),'checkpoint':str(asset/'ckpts/ckpt_4999.pt')}
    elif (R/'demo_jobs'/f'{label}.json').exists():
        job=read(R/'demo_jobs'/f'{label}.json');source=Path(job['source']);original=source.parents[1]
        asset=Path(job['checkpoint']).parents[1];demo=Path(job['output']);mapping=Path(job['mapping'])
        if label in SHORT:original=Path(job['native_mapping']).parent
    else:
        entry=next(x for x in read(R/'next_demos_plan.json')['jobs'] if x['id']==label)
        original=Path(entry['source']);source=Path(entry['output'])/'outputs/smoke'
        asset=None;demo=None;mapping=original/'inputs/frame_mapping.json'
        if read(source/'preview/final_validation.json')['state']!='validated':raise ValueError('Unvalidated generation')
        job={'source':str(source),'input':str(original/'inputs/main.mp4'),'mapping':str(mapping),
             'native_mapping':str(original/'inputs_native/frame_mapping.json')}
    m=read(mapping);candidate=original/'candidate.json';c=read(candidate) if candidate.exists() else {}
    # Never infer cross-scene or cross-camera identity. Legacy phone input has
    # no segment field in this manifest, so retain its scoped camera-track key.
    identity=m.get('identity_key') or c.get('identity_key')
    if not identity:identity=f"legacy:{m.get('source_split','train')}:{m['scene']}:cam{m['camera_index']}:{m['track_id']}"
    action='行走（root-local）' if walking else '动作待复核'
    baseline=read(R/'reviews/baseline12.json')['items']
    if label in baseline:action=baseline[label]['action']
    if label.startswith('batch_') and label.endswith('_rootlocal24'):
        action=next(x['review'] for x in read(R/'next_demos_plan.json')['jobs'] if Path(x['source']).name==label.removesuffix('_rootlocal24'))
    if asset is None:action=next(x['review'] for x in read(R/'next_demos_plan.json')['jobs'] if x['id']==label)
    ego=m.get('ego_translation_path_m',c.get('ego_path_m'))
    if label in SHORT:
        action='行走／2.4秒短片段' if label=='r07_short61_24' else '行走／持续移动自车'
        camera_key='camera_exposure_center_path_m' if label=='r07_short61_24' else 'camera_path_m'
        ego=read(original/'research/camera_motion.json')[camera_key]
        native=read(Path(job['native_mapping']))
        m=dict(m,real_source_frame_range=[native['frames'][0]['frame_index'],native['frames'][-1]['frame_index']])
    state='技术完成；实验demo；成品质量未批准' if asset else '六视角生成完成；尚无高斯资产'
    if label in baseline and baseline[label]['issue_category']=='generation_failure':state='技术完成；明确生成外观失败对照'
    if label=='batch_0014_rootlocal6_v1':state='六视角完成；附件生成失败对照；不升级'
    if label=='m02_short61_24':state='技术完成；重建视觉失败：多个时刻小腿脚部模糊或缺失'
    items=[]
    def add(stage,name,target,role,run='shared'):
        p=Path(target)
        if p.is_file():items.append({'link':stage+'/'+name,'target':str(p.absolute()),'role':role,'run':run,
                                    'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
    add('01_input','native_mapping.json',job['native_mapping'],'原始帧、真实时间与来源引用')
    add('01_input','candidate.json',candidate,'身份、自车运动与筛选依据')
    frames=m['frames']
    for tag,index in [('first',0),('middle',len(frames)//2),('last',len(frames)-1)]:
        source_path=frames[index].get('source_path')
        if source_path:add('01_input',tag+'.jpg',source_path,'原始驾驶全景；目标框见frame_mapping')
    add('02_preprocess','input.mp4',job['input'],'实际模型人物输入')
    add('02_preprocess','frame_mapping.json',mapping,'裁剪、真实/插值时刻、相机变换')
    add('02_preprocess','target_masks.npz',original/'segmentation/target_masks.npz','实际指定人物掩码')
    add('02_preprocess','mask_review.jpg',original/'previews/foreground_review.jpg','源图与分割抽样')
    add('02_preprocess','camera_motion.json',original/'research/camera_motion.json','实际所选时间窗的曝光中心相机运动')
    motion_root=source.parents[1]/'preflight'
    for name in ('motion/motion.json','motion/motion.safetensors','root_motion.npz','conditioning/metadata.json'):
        add('03_motion',name.replace('/','_'),motion_root/name,'生成实际采用的运动/构图')
    for name in ('cameras.json','metadata.json','root_motion.npz','root_local.json','preview/multiview.mp4','preview/time_views.jpg','preview/final_validation.json'):
        add('04_generation',name.replace('/','_'),source/name,'实际生成产物')
    add('04_generation','short24_validation.json',source/'preview/short24_validation.json','61帧24视角独立解码与精确时间验收')
    for p in sorted((source/'videos').glob('*.mp4')):add('04_generation','view_'+p.name,p,'原始生成视角')
    selected='q0' if asset else 'six_view'
    selected_reason='先看完整基线链路；不是最终视觉批准。'
    if asset:
        if read(asset/'preview/validation.json')['status']!='trained_rendered_exported_validated':raise ValueError('Unvalidated asset')
        add('05_reconstruction','q0_cfg.yml',asset/'cfg.yml','原5k训练配置','q0')
        add('05_reconstruction','q0_checkpoint.pt',job['checkpoint'],'原5k动态检查点','q0')
        add('05_reconstruction','init.npz',Path(job['data'])/'init.npz','原训练初始化','q0')
        add('06_assets','q0.ftgs.ply',asset/'ckpt_4999.ftgs.ply','完整未滤点动态资产','q0')
        add('06_assets','asset_q0.json',asset/'asset.json','5k资产坐标与时间说明','q0')
        add('06_assets','q0_validation.json',asset/'preview/validation.json','5k导出及解码验收','q0')
        for name,outname in [('four_panel','01_pipeline'),('fixed_views','02_motion_fixed_views'),
                             ('orbit_fixed_time','03_geometry_orbit'),('replay_original','05_original_root_replay'),
                             ('replay_new','06_synthetic_path_replay')]:
            add('07_demo',outname+'.mp4',demo/(name+'.mp4'),'5k主展示；原/生成相机不同；轨迹为估计或合成','q0')
        add('08_evaluation','q0_metrics.json',R/'benchmark/results'/f'{label}_q0/metrics.json','全121帧整图白背景协议','q0')
        add('08_evaluation','q0_sampled_metrics.json',asset/'quality_metrics/metrics.json','3个留出生成视角×13时刻×3背景；真实新视角无真值','q0')
        for run in ('budget_half_50k','capacity50k'):
            root=R/'quality'/f'{label}_{run}';metric=R/'benchmark/results'/f'{label}_{run}/metrics.json'
            if metric.exists() and read(metric)['state']=='complete':
                add('05_reconstruction',run+'_checkpoint.pt',root/'ckpts/ckpt_49999.pt','50k版本检查点',run)
                add('05_reconstruction',run+'_cfg.yml',root/'cfg.yml','50k版本训练配置',run)
                add('06_assets',run+'.ftgs.ply',root/'ckpt_49999.ftgs.ply','50k完整动态资产；坐标沿用同源数据',run)
                add('08_evaluation',run+'_metrics.json',metric,'同源同协议比较',run)
                if run=='capacity50k':selected=run;selected_reason='相同50k预算下已有整图指标改善；建议配合对照视频看，仍未批准生产质量。'
        for suffix in ('_validation','_optimization',''):
            comparison=R/'comparisons'/(label+suffix)/'view08.mp4'
            add('08_evaluation','comparison'+(suffix or '_budget')+'.mp4',comparison,'同生成相机/时间的版本对照','comparison')
    add('08_evaluation','protocol.json',R/'benchmark/protocol.json','整图指标协议')
    add('08_evaluation','generation_review.json',R/'reviews'/(label+'_24.json'),'生成阶段抽样复核与重建许可')
    add('08_evaluation','asset_review.json',R/'reviews'/(label+'_asset.json'),'动态重建抽样复核与限制')
    if walking:
        add('08_evaluation','motion_details.mp4',R/'diagnostics/walking_temporal/motion_details.mp4','同相机逐时间头/手/脚细节：生成、5k、原50k、增密50k','comparison')
        add('08_evaluation','temporal_diagnostic.json',R/'diagnostics/walking_temporal/diagnostic.json','逐时间指标与参数分布诊断，不是因果证明','comparison')
    if label=='r07_short61_24':
        diagnostic=R/'diagnostics/weak_velocity'
        add('07_demo','15k_zero_sigma8_pipeline.mp4',diagnostic/'zero_sigma8/demo/four_panel.mp4','15k候选优化版本四联屏，仍模糊','zero_sigma8')
        add('08_evaluation','initialization_comparison.mp4',diagnostic/'comparison_sigma8/view08.mp4','四组15k初始化对照，含明确负结果','comparison')
        add('08_evaluation','initialization_report.md',diagnostic/'REPORT.md','相同预算的速度和初始持续时间对照','comparison')
        add('08_evaluation','before_after.mp4',diagnostic/'before_after/view08.mp4','生成/原5k/普通15k/初始sigma8直接对照','comparison')
        for variant in ('zero','body','zero_sigma8','body_sigma8'):
            add('06_assets',variant+'_15k.ftgs.ply',diagnostic/variant/'ckpt_14999.ftgs.ply','诊断版本，非新增人物，非生产批准',variant)
            add('06_assets',variant+'_asset.json',diagnostic/variant/'asset.json','版本时间/坐标及质量说明',variant)
    if label=='m02_short61_24':
        diagnostic=R/'diagnostics/m02_duration'
        add('08_evaluation','duration_before_after.mp4',diagnostic/'comparison/view00.mp4','生成/原5k/普通15k/初始sigma8直接对照','comparison')
        add('08_evaluation','duration_results.json',diagnostic/'results.json','M02同15k预算、相同点数的初始时间宽度对照','comparison')
        add('08_evaluation','duration_review.json',diagnostic/'comparison/review.json','下肢改善但脚部仍淡化；抽样范围有限','comparison')
        for variant in ('sigma15','sigma8'):
            add('06_assets',variant+'_15k.ftgs.ply',diagnostic/variant/'ckpt_14999.ftgs.ply','15k诊断版本；非新增人物，非生产批准',variant)
            add('06_assets',variant+'_asset.json',diagnostic/variant/'asset.json','15k诊断版本坐标/时间与文件哈希',variant)
            for name in ('four_panel','fixed_views','orbit_fixed_time','backgrounds'):
                add('07_demo',f'15k_{variant}_{name}.mp4',diagnostic/variant/'demo'/f'{name}.mp4','15k诊断展示，仍未视觉批准',variant)
            add('08_evaluation',variant+'_demo_validation.json',diagnostic/variant/'demo/validation.json','完整动作/环绕解码与输入来源记录',variant)
    for version in completed_variants(label):
        run=version['run'];root=version['root'];info=version['asset'];tag='4dgs_'+run
        add('05_reconstruction',tag+'_config.json',root/'experiment.json','新后端实际初始化/损失/预算，逐资产优化',run)
        add('05_reconstruction',tag+'_checkpoint.pth',root/'chkpnt_fine_30000.pth','新后端终态优化器检查点',run)
        for key in ('canonical_ply','deformation_weights','deformation_table','deformation_accum'):
            p=root/info[key]
            add('06_assets',tag+'_'+p.name,p,'4DGS动态包组件；仅规范PLY不能动态重放',run)
        add('06_assets',tag+'_asset.json',root/'asset.json','新后端动态包及坐标/时间/哈希',run)
        add('08_evaluation',tag+'_metrics.json',root/'evaluation/results.json','同生成观测留出指标和新进程重放',run)
        add('08_evaluation',tag+'_comparison.mp4',root/'ftgs_comparison/view08.mp4','与FTGS同相机/时间直接对照',run)
        if version['validated_demo']:
            for name in ('four_panel','fixed_views','orbit_fixed_time','backgrounds'):
                add('07_demo',tag+'_'+name+'.mp4',root/'demo'/f'{name}.mp4','4DGS完整未滤点候选展示；未生产批准',run)
            add('08_evaluation',tag+'_demo_validation.json',root/'demo/validation.json','新后端动作、环绕与源观测时间验收',run)
    add('09_records','code_manifest.json',R/'code_manifest.json','代码版本快照')
    add('09_records','source_job.json',R/'demo_jobs'/f'{label}.json','原路径和依赖')
    return {'label':label,'identity_key':identity,'identity_scope':'Do not merge unverified cross-camera/cross-scene identities',
            'scene':m['scene'],'track':m['track_id'],'camera':m['camera_index'],'action':action,
            'ego':'近静止' if ego is not None and ego<.1 else ('明确移动' if ego is not None and ego>1 else '近静止基线，数值见源审计'),
            'ego_path_m':ego,'state':state,'source_frame_range':m.get('real_source_frame_range'),
            'real_frames':m.get('real_source_frames'),'output_frames':m.get('output_frames'),
            'selected_run':selected,'selected_reason':selected_reason,'production_approved':False,
            'source_generation':str(source),'source_data':job.get('data'),'artifacts':items}

def page(row,oid):
    esc=html.escape;clip='clips/C001/';art=row['artifacts']
    videos=[x for x in art if x['link'].startswith('07_demo/') and Path(x['link']).name=='01_pipeline.mp4']
    if not videos:videos=[x for x in art if x['link'].startswith('04_generation/') and Path(x['link']).name=='preview_multiview.mp4']
    main='' if not videos else f'<video controls preload="metadata" src="{clip+videos[0]["link"]}"></video>'
    parts=[f'<!doctype html><meta charset="utf-8"><title>{oid}</title><style>body{{font:16px sans-serif;max-width:1100px;margin:30px auto;line-height:1.6}}video{{width:100%;max-height:650px}}details{{margin:20px 0}}code{{overflow-wrap:anywhere}}</style>',
           f'<a href="../index.html">全部对象</a><h1>{oid} · {esc(row["label"])}</h1><p>{esc(row["ego"])}自车 · {esc(row["action"])}</p>',
           f'<p>{esc(row["state"])}。真实帧 {row["real_frames"]}，输出时间槽 {row["output_frames"]}；源帧范围 {esc(str(row["source_frame_range"]))}。</p>',
           f'<p>建议比较版本：{esc(row["selected_run"])}。{esc(row["selected_reason"])}</p>',
           '<p>下方主展示如有高斯为原5k；50k比较在评价阶段。推荐观看不等于生产批准。原始相机与生成相机不同。</p>',main]
    for stage in STAGES:
        entries=[x for x in art if x['link'].startswith(stage+'/')]
        parts.append(f'<details><summary>{stage}</summary><ul>')
        parts.extend(f'<li><a href="{clip+x["link"]}">{esc(x["role"])} · {esc(Path(x["link"]).name)}</a><br><code>{esc(x["target"])}</code></li>' for x in entries)
        parts.append('</ul></details>')
    return '\n'.join(parts)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');parser.add_argument('--labels',nargs='*');args=parser.parse_args()
    labels=args.labels or ['phone_full','walk063_rootlocal24']+[x['label'] for x in read(W/'batch100/closeout12.json')['assets'] if x['label']!='phone_full']
    if not args.labels:
        for entry in read(R/'next_demos_plan.json')['jobs']:
            source_name=Path(entry['source']).name
            asset_label=source_name+'_rootlocal24'
            complete=R/'reconstruction/outputs'/asset_label/'preview/validation.json'
            if complete.exists() and read(complete)['status']=='trained_rendered_exported_validated' and (R/'demo_jobs'/f'{asset_label}.json').exists():
                labels.append(asset_label)
            elif (Path(entry['output'])/'outputs/smoke/preview/final_validation.json').exists():
                labels.append(entry['id'])
        for label in sorted(SHORT):
            short=R/'reconstruction/outputs'/label/'preview/validation.json'
            if short.exists() and read(short)['status']=='trained_rendered_exported_validated':labels.append(label)
    O.mkdir(exist_ok=True)
    with (O/'.publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registry=read(O/'registry.json') if (O/'registry.json').exists() else {}
        rows=[record(label) for label in labels]
        for row in rows:
            key=row['identity_key']
            if key not in registry:registry[key]=f'OBJ_{len(registry)+1:04d}'
            row['object_id']=registry[key]
            clip=O/row['object_id']/'clips/C001'
            for item in row['artifacts']:
                dest=clip/item['link']
                if dest.is_symlink() and dest.resolve()!=Path(item['target']).resolve():
                    # Keep the old version's link intact. New versions receive
                    # explicit run paths rather than silently retargeting it.
                    stage,name=item['link'].split('/',1)
                    item['link']=f'{stage}/runs/{row["label"]}/{name}'
        if len({x['identity_key'] for x in rows})!=len(rows):raise ValueError('Multiple clips require explicit grouping')
        plan={'mode':'apply' if args.apply else 'dry_run','objects':rows,'data_copy_bytes':0,'source_mutation':False}
        atomic(O/'mapping_plan.json',jsontext(plan))
        if not args.apply:print(jsontext({'mode':'dry_run','objects':len(rows),'links':sum(len(x['artifacts']) for x in rows)}));return
        inventory=[];checks=[]
        def publish(path,text):
            if path.is_symlink():raise ValueError('Refuse writing through a link')
            atomic(path,text);inventory.append(str(path.relative_to(O)))
        for row in rows:
            obj=O/row['object_id'];clip=obj/'clips/C001';clip.mkdir(parents=True,exist_ok=True)
            for stage in STAGES:(clip/stage).mkdir(exist_ok=True)
            for item in row['artifacts']:
                dest=clip/item['link'];source=Path(item['target']);before=source.stat()
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.is_symlink():
                    if dest.resolve()!=source.resolve():raise ValueError('Existing link points to another version')
                elif dest.exists():raise FileExistsError(dest)
                else:dest.symlink_to(os.path.relpath(source,dest.parent))
                after=source.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during publication')
                checks.append({'link':str(dest.relative_to(O)),'resolved':str(dest.resolve()),'source_stat_unchanged':True})
                inventory.append(str(dest.relative_to(O)))
            previous=clip/'manifest.json'
            if previous.exists():
                old=read(previous)
                if old['label']!=row['label']:
                    archive=clip/'09_records'/('manifest_'+old['label']+'.json')
                    if not archive.exists():publish(archive,jsontext(old))
            publish(obj/'object.json',jsontext({k:v for k,v in row.items() if k!='artifacts'}))
            publish(clip/'manifest.json',jsontext({'object_id':row['object_id'],'clip':'C001','label':row['label'],'artifacts':row['artifacts'],
                'dependency_chain':{'source':row['identity_key'],'input':next(a['target'] for a in row['artifacts'] if a['link'].startswith('02_preprocess/') and Path(a['link']).name=='input.mp4'),
                    'motion':str(Path(row['source_generation']).parents[1]/'preflight/motion'),'generation':row['source_generation'],
                    'reconstruction_data':row['source_data']},'unprovided_standard_files':['04_motion_details.mp4: existing still details only'],
                'scope':'Read-only browse links; relative symlinks are not portable standalone downloads.'}))
            publish(clip/'selected_run.json',jsontext({'run':row['selected_run'],'reason':row['selected_reason'],'production_approved':False,
                'primary_pipeline_video_run':'q0' if row['source_data'] else 'six_view','comparison_stage':'08_evaluation'}))
            description=f'# {row["object_id"]} · {row["label"]}\n\n{row["ego"]}自车；{row["action"]}。{row["state"]}。\n\n先打开 [对象页面](index.html)。原图、实际输入、运动、生成、重建和资产依次见 clips/C001/01_input 至 09_records。\n\n建议版本：{row["selected_run"]}。{row["selected_reason"]} 主四联屏仍为5k，50k放在比较页，不混用版本。\n\n文件均为引用，原件不搬动、不复制；请勿通过链接修改源文件。下载此目录到其他机器时链接不会自动携带数据。\n'
            publish(obj/'README.md',description);publish(clip/'README.md',description.replace('(index.html)','(../../index.html)'))
            publish(obj/'index.html',page(row,row['object_id']))
        publish(O/'registry.json',jsontext(registry));publish(O/'catalog.json',jsontext({'updated_at':datetime.now(timezone.utc).isoformat(),'objects':[{k:v for k,v in row.items() if k!='artifacts'} for row in rows]}))
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=['object_id','label','scene','track','ego','action','state','selected_run']);writer.writeheader();writer.writerows({k:x[k] for k in writer.fieldnames} for x in rows);publish(O/'catalog.csv',buffer.getvalue())
        cards='\n'.join(f'<li data-text="{html.escape(x["ego"]+x["action"]+x["state"],quote=True)}"><a href="{x["object_id"]}/index.html">{x["object_id"]} · {html.escape(x["label"])}</a> — {html.escape(x["action"])}；{html.escape(x["state"])}</li>' for x in rows)
        publish(O/'index.html','<!doctype html><meta charset="utf-8"><title>Waymo 人体对象</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;line-height:1.7}li{margin:16px 0}</style><h1>Waymo → 4DAnyone → 动态高斯</h1><p>优先看 OBJ_0002 行走与版本对照；OBJ_0001 是站立参照。各阶段点击展开。实验demo与生产批准分开。</p><input id="filter" placeholder="筛选：行走、近静止、失败…"><ul>'+cards+'</ul><script>document.querySelector("#filter").oninput=e=>document.querySelectorAll("li").forEach(x=>x.hidden=!x.textContent.includes(e.target.value));</script>')
        publish(O/'validation.json',jsontext({'state':'links_and_source_stats_verified','objects':len(rows),'links':len(checks),'checks':checks,
            'source_data_copied':False,'source_files_changed_by_publisher':False,'browser_playback_test':'not_performed; existing source video decode proofs reused',
            'export_roundtrip':'Existing per-asset validation linked, not rerun by organizer'}))
        publish(O/'publication_inventory.json',jsontext({'managed_entries':inventory,'rollback':'Remove only these relative entries with unlink; never resolve symlinks for deletion. Do not remove pre-existing source artifacts.'}))
        print(jsontext({'mode':'published','objects':len(rows),'links':len(checks),'index':str(O/'index.html')}))

if __name__=='__main__':main()
