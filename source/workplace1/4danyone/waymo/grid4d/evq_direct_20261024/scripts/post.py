"""Publish terminal-only evidence. Numerical success never implies visual pass."""
import argparse,asyncio,csv,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np

N=Path(__file__).resolve().parents[1];ROOT=N.parent;OLD=ROOT/'early_visual_query_20261019'
BASE=ROOT.parent/'random8/r20261015_113332'
spec=importlib.util.spec_from_file_location('fixed_aggregation',ROOT/'optimization_quality_v3_20261018/scripts/analyze_quality.py')
aggregation=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregation)
FIELDS=aggregation.FIELDS
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvrows(p):return list(csv.DictReader(Path(p).open()))
def write(p,rows):
    with Path(p).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze():
    plan=read(N/'PLAN.json');metrics={};parts={};rows=[];gates=[];effects=[];assets=[];costs=[]
    for label in plan['labels']:
        signature=None
        for method in ['A','R0','R3']+plan['arms']:
            if method=='A':
                ref=plan['inputs'][label]['old_references']['A'];mp=Path(ref['metrics']);pp=Path(ref['parts'])
            else:
                folder=(OLD if method in ('R0','R3') else N)/'results'/f'{label}_{method}_32999';mp=folder/'metrics.json';pp=folder/'parts.csv'
            metric=read(mp);pr=csvrows(pp)
            current=[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in pr]
            if signature is None:signature=current
            else:assert current==signature,(label,method,'part reference denominators changed')
            metrics[label,method]=metric['mean'];parts[label,method]=aggregation.part_summary(pr)
            rows.append(dict(label=label,method=method,**{k:metric['mean'][k] for k in FIELDS}))
        if label in ('N06','N11'):
            previous=ROOT/'evq_primitive_local_20261019/results'/f'{label}_G2_32999'
            metric=read(previous/'metrics.json');pr=csvrows(previous/'parts.csv')
            assert [(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in pr]==signature
            metrics[label,'G2']=metric['mean'];parts[label,'G2']=aggregation.part_summary(pr)
            rows.append(dict(label=label,method='G2',**{k:metric['mean'][k] for k in FIELDS}))
        for arm in plan['arms']:
            key=f'{label}_{arm}';asset=N/'assets'/key
            assert read(asset/'replay_validation.json')['passed']
            done=read(N/'runs'/key/'training.complete.json')
            assert done['step']==32999 and done['counts']['additional_updates']==30000
            assets.append(dict(key=key,asset=str(asset),recovery=str(N/'runs'/key/'recovery_32999.pt'),actual_updates=32999,replay_passed=True))
            costs.append(dict(key=key,additional_updates=30000,logical_updates=32999,train_entry_seconds=done['seconds'],loop_seconds=done['loop_seconds'],load_seconds=done['load_seconds'],visibility_renders=done['counts'].get('visibility_renders',0),target_renders=done['counts']['target_renders'],alpha_renders=done['counts']['alpha_renders'],gpu_uuid=done['device']['uuid'],points=done['points']))
            for comparator in ['A','R0','R3','P1','P2','P3']+(['G2'] if label!='N04' else []):
                if arm==comparator:continue
                a,b=metrics[label,arm],metrics[label,comparator]
                checks=[aggregation.comparison_check(k,rule,tol,a[k],b[k]) for k,rule,tol in [('foreground_psnr','drop_max',.2),('ssim','drop_max',.001),('crop_lpips','increase_max',.002),('outside_alpha_mean','ratio_max',1.05)]]
                for part in aggregation.PARTS:
                    pa,pb=parts[label,arm][part],parts[label,comparator][part]
                    checks += [aggregation.comparison_check('coverage_'+part,'drop_max',.01,pa['coverage'],pb['coverage']),aggregation.comparison_check('temporal_'+part,'ratio_max',1.05,pa['compensated_rmse'],pb['compensated_rmse'])]
                gates.append(dict(label=label,candidate=arm,comparator=comparator,checks=checks,numeric_pass=all(c['state']=='pass' for c in checks),visual_review='pending'))
        if label!='N04':
            for candidate,comparator in [('P3','P2'),('P3','P1'),('P4','P3')]:
                ca=read(N/'results'/f'{label}_{candidate}_32999/expanded.json')['summary']
                cb=read(N/'results'/f'{label}_{comparator}_32999/expanded.json')['summary']
                for scope in ('side0','side1','fixed_bad_window_left'):
                    a,b=ca[scope],cb[scope]
                    assert (a['known_rows'],a['roi_pixels'],a['reference_alpha_mass'])==(b['known_rows'],b['roi_pixels'],b['reference_alpha_mass'])
                    for route,elimit,dlimit in [('appearance',.95,1.),('completeness',1.,.9)]:
                        checks=[aggregation.comparison_check('RGB_MSE','ratio_max',elimit,a['RGB_MSE'],b['RGB_MSE']),aggregation.comparison_check('undercoverage','ratio_max',dlimit,a['soft_undercoverage'],b['soft_undercoverage'])]
                        effects.append(dict(label=label,candidate=candidate,comparator=comparator,scope=scope,route=route,role='descriptive_capacity_no_extra_gate' if comparator=='P1' else 'registered_effectiveness',checks=checks,known_rows=a['known_rows'],unknown_rows=a['unknown_rows'],measured_numeric_pass=all(c['state']=='pass' for c in checks),full_numeric_pass=all(c['state']=='pass' for c in checks) and a['unknown_rows']==0,visual_review='pending'))
    means=[dict(method=method,people=3,**{k:float(np.mean([metrics[label,method][k] for label in plan['labels']])) for k in FIELDS}) for method in ['A','R0','R3']+plan['arms']]
    write(N/'paired_metrics.csv',rows);write(N/'means.csv',means);write(N/'costs.csv',costs)
    two=[dict(method=m,people=2,labels='N06,N11',**{k:float(np.mean([metrics[l,m][k] for l in ['N06','N11']])) for k in FIELDS}) for m in ['A','R3','G2']+plan['arms']]
    write(N/'two_person_means.csv',two)
    save(N/'gate_results.json',dict(protection=gates,effectiveness=effects,unknown_not_pass=True,default_changed=False))
    save(N/'ASSETS.json',assets)
    lines=['Direct Gaussian Query：数值结果（人工视觉复核尚未完成）','三人等权；参考是冻结4DAnyone生成H开发观测，不是真实Waymo新视角真值。','A为52999次单目标/后期alpha历史配方；R0/R3/P1-P4为32999次双目标协议；训练时间不能只按次数比较。','', 'method,foreground_PSNR,SSIM,crop_LPIPS,outside_alpha']
    lines += [f"{r['method']},{r['foreground_psnr']:.6f},{r['ssim']:.6f},{r['crop_lpips']:.6f},{r['outside_alpha_mean']:.6f}" for r in means]
    lines += ['', '完整更新与时间见costs.csv，逐人保护线见gate_results.json。数值通过不等于视觉质量通过，不自动替换完整A。']
    (N/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    return means


def media(label):
    from PIL import Image,ImageDraw
    import cv2
    sys.path.insert(0,str(BASE/'backend/src'))
    from video_io import MP4Writer
    cv2.setNumThreads(2)
    plan=read(N/'PLAN.json');count=plan['inputs'][label]['frame_count'];folder=N/'comparison'/label
    folder.mkdir(exist_ok=False)
    # Final complete comparisons only. Each row remains large enough to inspect.
    names=['reference','A','R3']+(['G2'] if label!='N04' else [])+['P1','P2','P3','P4']
    paths={a:N/'cache/raw'/f'{label}_{a}_32999' for a in plan['arms']}
    paths['A']=Path(plan['inputs'][label]['old_references']['A']['raw']);paths['R3']=OLD/'cache/raw'/f'{label}_R3_32999'
    if label!='N04':paths['G2']=ROOT/'evq_primitive_local_20261019/cache/raw'/f'{label}_G2_32999'
    receipts=[]
    for view in (0,8,16):
        arrays={}
        for method,path in paths.items():
            with np.load(path/f'view{view:02d}.npz') as x:arrays[method]=(x['rgb'],x['alpha'])
        path=folder/f'view{view:02d}.mp4';contact=Image.new('RGB',(352*len(names),672*3),'#222222')
        with MP4Writer(str(path),25,352*len(names),672) as writer:
            for frame in range(count):
                rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
                reference=rgba[:,:,:3]*rgba[:,:,3:]+.5*(1-rgba[:,:,3:])
                images=[reference if method=='reference' else np.clip(arrays[method][0][frame]+.5*(1-arrays[method][1][frame,...,None]),0,1) for method in names]
                canvas=Image.new('RGB',(352*len(names),672),'#222222');canvas.paste(Image.fromarray(np.rint(np.concatenate(images,1)*255).astype(np.uint8)),(0,32));draw=ImageDraw.Draw(canvas)
                for col,method in enumerate(names):draw.text((352*col+5,8),f'{label} {method} v{view:02d} f{frame}',fill='white')
                writer.append_data(np.asarray(canvas))
                if frame in (0,(count-1)//2,count-1):contact.paste(canvas,(0,672*[0,(count-1)//2,count-1].index(frame)))
        contact.save(folder/f'view{view:02d}.jpg',quality=94)
        cap=cv2.VideoCapture(str(path));decoded=0
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==(672,352*len(names),3);decoded+=1
        cap.release();assert decoded==count
        receipts.append(dict(path=str(path.relative_to(N)),decoded_frames=decoded,columns=names))
        if label in ('N06','N11'):
            bad=plan['evaluation']['fixed_bad_windows'][label]
            if view==bad['view']:
                times=[bad['frames'][0],sum(bad['frames'])//2,bad['frames'][1]]
                if label=='N11':times.append(count-1)
                for frame in times:
                    rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
                    ref=rgba[:,:,:3]*rgba[:,:,3:]+.5*(1-rgba[:,:,3:])
                    images=[ref if m=='reference' else np.clip(arrays[m][0][frame]+.5*(1-arrays[m][1][frame,...,None]),0,1) for m in names]
                    Image.fromarray(np.rint(np.concatenate(images,1)*255).astype(np.uint8)).save(folder/f'fixed_v{view:02d}_f{frame:03d}.jpg',quality=95)
    save(folder/'complete.json',dict(fully_decoded=True,files=receipts,visual_review='pending'))


async def main():
    assert read(N/'pipeline_queue.complete.json')['all_passed']
    means=analyze();sem=asyncio.Semaphore(2)
    async def render(label):
        async with sem:
            with (N/'logs'/f'media_{label}.log').open('a') as log:
                p=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__)),'--media',label,stdout=log,stderr=log)
                assert await p.wait()==0,label
    await asyncio.gather(*(render(l) for l in ('N04','N06','N11')))
    html=['<!doctype html><meta charset="utf-8"><title>Direct Gaussian Query</title><style>body{font-family:sans-serif;background:#191919;color:#eee;max-width:1800px;margin:auto}video,img{width:100%}a{color:#9cf}td,th{padding:8px}</style><h1>高斯直接 Query：完整终态对照</h1><p>人工动态视觉复核待完成。H为生成开发参考；禁止将数值通过当作真实人体重建精度。</p><a href="../RESULTS.txt">数值说明</a> · <a href="../means.csv">均值表</a> · <a href="../ASSETS.json">动态资产</a><table><tr><th>方法</th><th>前景PSNR↑</th><th>SSIM↑</th><th>LPIPS↓</th></tr>']
    html += [f"<tr><td>{r['method']}</td><td>{r['foreground_psnr']:.4f}</td><td>{r['ssim']:.5f}</td><td>{r['crop_lpips']:.5f}</td></tr>" for r in means];html.append('</table>')
    for label in ('N04','N06','N11'):
        html.append(f'<h2>{label}</h2><p>列：生成参考 / A53k / 旧锚点R3 / （N06、N11另含旧局部外观G2）/ P1容量 / P2固定读取 / P3直接Query / P4可见性Query</p>')
        for v in (0,8,16):html.append(f'<video controls preload="metadata" src="{label}/view{v:02d}.mp4"></video><a href="{label}/view{v:02d}.jpg">首中末图 v{v:02d}</a>')
    (N/'comparison/index.html').write_text('\n'.join(html))
    save(N/'post.complete.json',dict(numeric_media_complete=True,visual_review='pending',default_changed=False,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--media');args=parser.parse_args()
    if args.media:media(args.media)
    else:asyncio.run(main())
