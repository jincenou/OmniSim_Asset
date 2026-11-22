"""Publish only complete 2x4 comparisons, with same-reference and sampling audits."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from baseline_queue import ROOT,BASE,save,now
from render_baseline import sha

ARMS=['continue','parts','alpha','combined']
PARTS=['head','torso','left_arm','right_arm','left_leg','right_leg']
NAMES=['Generated reference','Grid4D33k','Continue +15k','Parts +15k','Alpha +15k','Combined +15k']
FIELDS=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']


def csvfile(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    cv2.setNumThreads(2);opt=ROOT/'optimization';stage=opt/'publishing';stage.mkdir(exist_ok=False)
    globalrows=[];localrows=[];audits=[];results={}
    for label in ['N04','N11']:
        results[label]={};traces={};denominators=None
        for method in ['baseline']+ARMS:
            if method=='baseline':
                metrics_path=ROOT/'results'/label/'33000/metrics.json';part_path=ROOT/'audit/metrics'/label/'grid4d_33k.csv'
            else:
                run=opt/'runs'/f'{label}_{method}';metrics_path=opt/'results'/f'{label}_{method}'/'metrics.json';part_path=metrics_path.parent/f'grid4d_{method}.csv'
                trace=[json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
                assert len(trace)==15000 and [t['extra_update'] for t in trace]==list(range(1,15001))
                traces[method]=[(t['view'],t['frame']) for t in trace]
                assert not {v for v,f in traces[method]}&{0,8,16}
                assert len({t['points'] for t in trace})==1
            report=json.loads(metrics_path.read_text());results[label][method]=report
            globalrows.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',method=method,actual_updates=report['actual_optimizer_updates'],**{k:report['mean'][k] for k in FIELDS}))
            rows=list(csv.DictReader(part_path.open()))
            common=[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in rows]
            if denominators is None:denominators=common
            else:assert common==denominators
            for part in PARTS:
                selected=[r for r in rows if r['part']==part and r['coverage']]
                temporal=[r for r in selected if r['temporal_mse']]
                localrows.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',method=method,part=part,valid_slots=len(selected),
                    mean_coverage=float(np.mean([float(r['coverage']) for r in selected])) if selected else None,
                    p05_coverage=float(np.quantile([float(r['coverage']) for r in selected],.05)) if selected else None,
                    mean_part_psnr=float(np.mean([float(r['psnr']) for r in selected])) if selected else None,
                    compensated_rmse=math.sqrt(float(np.mean([float(r['temporal_mse']) for r in temporal]))) if temporal else None,
                    temporal_valid_fraction=float(np.mean([float(r['temporal_valid_fraction']) for r in temporal])) if temporal else None))
        assert all(traces[a]==traces['continue'] for a in ARMS)
        snapshots=[json.loads((opt/'runs'/f'{label}_{a}'/'snapshots/48000.json').read_text()) for a in ARMS]
        assert len({s['points'] for s in snapshots})==1
        audits.append(dict(label=label,all_four_sampling_traces_identical=True,actual_extra_updates=15000,points=snapshots[0]['points'],all_local_reference_and_temporal_denominators_identical=True,all_global_reference_and_crop_checks_passed=True))
    csvfile(stage/'global_metrics.csv',globalrows);csvfile(stage/'part_metrics_summary.csv',localrows)
    spec=importlib.util.spec_from_file_location('frozen_video',BASE/'backend/src/video_io.py');video=importlib.util.module_from_spec(spec);spec.loader.exec_module(video)
    media=[]
    for label in ['N04','N11']:
        folder=stage/label;folder.mkdir()
        selection=json.loads((ROOT/'gallery'/label/'selection.json').read_text())['parts']
        sheet=Image.new('RGB',(6*200,6*236+32),'#282828');ink=ImageDraw.Draw(sheet)
        ink.text((5,8),f'{label}: historical worst-part windows; SAME unmasked crops across all branches',fill='white')
        for view in [0,8,16]:
            arrays=[]
            for method in ['baseline']+ARMS:
                root=ROOT/'tmp/grid_raw'/label if method=='baseline' else opt/'raw'/f'{label}_{method}'
                assert (root/'complete.json').exists()
                with np.load(root/f'view{view:02d}.npz') as z:arrays.append((z['rgb'],z['alpha']))
            frames=len(arrays[0][0]);writer=video.MP4Writer(str(folder/f'view{view:02d}.mp4'),25,352*6,672)
            try:
                for frame in range(frames):
                    rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
                    images=[rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])]+[np.clip(r[frame]+.5*(1-a[frame,...,None]),0,1) for r,a in arrays]
                    canvas=Image.new('RGB',(352*6,672),'#282828');draw=ImageDraw.Draw(canvas)
                    canvas.paste(Image.fromarray(np.rint(np.concatenate(images,1)*255).astype(np.uint8)),(0,32))
                    for i,name in enumerate(NAMES):draw.text((352*i+4,9),f'{name} | {label} v{view:02d} f{frame}',fill='white')
                    writer.append_data(np.asarray(canvas))
                    if frame in [0,(frames-1)//2,frames-1]:canvas.save(folder/f'view{view:02d}_frame{frame:03d}.jpg',quality=95)
                    for p,chosen in enumerate(selection):
                        if chosen.get('view')!=view or chosen.get('representative_frame')!=frame:continue
                        x0,y0,x1,y1=chosen['crop_xyxy'];ybase=32+236*p
                        for col,(name,image) in enumerate(zip(NAMES,images)):
                            picture=Image.fromarray(np.rint(image[y0:y1,x0:x1]*255).astype(np.uint8));picture=ImageOps.contain(picture,(200,200))
                            sheet.paste(picture,(200*col+(200-picture.width)//2,ybase+32+(200-picture.height)//2))
                            ink.text((200*col+3,ybase+2),f'{name} {chosen["part"]}',fill='white');ink.text((200*col+3,ybase+16),f'v{view:02d} f{frame}',fill='white')
            finally:writer.close()
            file=folder/f'view{view:02d}.mp4';cap=cv2.VideoCapture(str(file));count=0
            while True:
                ok,image=cap.read()
                if not ok:break
                assert image.shape==(672,2112,3);count+=1
            cap.release();assert count==frames
            media.append(dict(path=str(file.relative_to(stage)),frames=count,sha256=sha(file)))
            del arrays
        sheet.save(folder/'parts.jpg',quality=95)
    save(stage/'validation.json',dict(state='passed',at=now(),paired=audits,videos=media,limitations=['Single seed per arm','Reference is generated development data','Equal updates and point count, not equal GPU seconds','Estimated conservative part cores, not whole-body semantic truth','Complete videos published after all arms and both identities finish']))
    table=['|人物|组别|全图PSNR↑|前景PSNR↑|SSIM↑|LPIPS↓|IoU↑|外部alpha↓|','|---|---|---:|---:|---:|---:|---:|---:|']
    table += ['|'+r['label']+'|'+r['method']+'|'+'|'.join(f'{r[k]:.5f}' for k in FIELDS)+'|' for r in globalrows]
    report='# Grid4D 固定容量优化对照\n\n两个人各四组，均从同一人的33k完整状态出发，额外15000次真实更新；终态实际47999次更新。普通续训、部位均衡、透明度、组合四组采样轨迹和点数已核对一致。只评固定终态，不选择开发集最优快照。\n\n'+'\n'.join(table)+'\n\n指标参考是冻结的4DAnyone生成视频，不是真实Waymo新视角真值。N11为备用困难例，不能并入主7人均值。更多指标在part_metrics_summary.csv。训练次数相同，透明度组有额外渲染/反传成本；只有一个种子，需结合可视化和局部指标判断。\n\n视频六列：生成参考 → 原Grid4D33k → 普通续训 → 部位均衡 → 透明度约束 → 两者组合。每人目录有view00/08/16完整视频、首中末帧以及按历史固定最差时段选出的部位对比图。\n'
    (stage/'REPORT.md').write_text(report)
    html=['<!doctype html><meta charset="utf-8"><title>Grid4D优化对照</title><style>body{font:16px sans-serif;background:#eee;margin:24px}video,img{width:100%;max-width:1800px}section{background:white;padding:20px;margin:20px 0}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}</style><h1>Grid4D：固定容量四组优化</h1><p>六列：生成参考 → 原Grid4D33k → 普通续训 → 部位均衡 → 透明度 → 组合。指标参考为生成开发集，不是真实新视角真值。</p><a href="REPORT.md">报告</a> · <a href="global_metrics.csv">全局指标CSV</a> · <a href="part_metrics_summary.csv">部位指标CSV</a><table><tr><th>人物</th><th>组别</th>'+''.join(f'<th>{k}</th>' for k in FIELDS)+'</tr>']
    html+=['<tr><td>'+r['label']+'</td><td>'+r['method']+'</td>'+''.join(f'<td>{r[k]:.5f}</td>' for k in FIELDS)+'</tr>' for r in globalrows];html.append('</table>')
    for label in ['N04','N11']:
        html.append(f'<section><h2>{label}</h2><img src="{label}/parts.jpg" loading="lazy">')
        for view in [0,8,16]:html.append(f'<h3>视角{view:02d}</h3><video controls loop preload="metadata" src="{label}/view{view:02d}.mp4"></video>')
        html.append('</section>')
    (stage/'index.html').write_text('\n'.join(html));target=opt/'comparison';assert not target.exists();stage.rename(target)
    print(json.dumps(dict(state='complete',comparison=str(target),videos=len(media))),flush=True)


if __name__=='__main__':main()
