"""Publish complete six-column comparisons only after all three terminal arms pass."""
import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;BASE=ROOT.parent/'random8/r20261015_113332'
sys.path.insert(0,str(ROOT/'scripts'))
from baseline_queue import save,now
from render_baseline import sha
FIELDS=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']
PARTS=['head','torso','left_arm','right_arm','left_leg','right_leg']
NAMES=['Generated reference','Common alpha47999','Historical combined','A alpha continue','U global stronger','B boundary stronger']


def csvfile(path,rows):
    with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',default='primary');args=parser.parse_args();cv2.setNumThreads(2)
    final=NEW/('comparison' if args.trial=='primary' else 'comparison_repeat');assert not final.exists()
    stage=NEW/f'publishing_{args.trial}';stage.mkdir(exist_ok=False);globalrows=[];localrows=[];audit=[]
    for label in ['N04','N11']:
        paths=[('HUST61',BASE/'samples'/label/'evaluations/4dgs_grid61/results.json',None),('Grid4D33k',ROOT/'results'/label/'33000/metrics.json',ROOT/'audit/metrics'/label/'grid4d_33k.csv')]
        paths += [(name,ROOT/'optimization/results'/f'{label}_{name}'/'metrics.json',ROOT/'optimization/results'/f'{label}_{name}'/f'grid4d_{name}.csv') for name in ['continue','parts','alpha','combined']]
        paths += [(name,NEW/'results'/args.trial/f'{label}_{name}'/'metrics.json',NEW/'results'/args.trial/f'{label}_{name}'/'parts.csv') for name in ['A','U','B']]
        denominators=None
        for name,path,partpath in paths:
            report=json.loads(path.read_text());globalrows.append(dict(label=label,method=name,trial=args.trial if name in ['A','U','B'] else 'historical',actual_updates=52999 if name in ['A','U','B'] else 47999 if name in ['continue','parts','alpha','combined'] else 32999,**{k:report['mean'][k] for k in FIELDS}))
            if partpath is None:continue
            rows=list(csv.DictReader(partpath.open()));current=[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in rows]
            if denominators is None:denominators=current
            else:assert current==denominators
            for part in PARTS:
                chosen=[r for r in rows if r['part']==part and r['coverage']!=''];times=[r for r in chosen if r['temporal_mse']!='']
                localrows.append(dict(label=label,method=name,part=part,valid_slots=len(chosen),mean_coverage=float(np.mean([float(r['coverage']) for r in chosen])) if chosen else None,mean_part_psnr=float(np.mean([float(r['psnr']) for r in chosen])) if chosen else None,compensated_rmse=math.sqrt(float(np.mean([float(r['temporal_mse']) for r in times]))) if times else None,temporal_valid_fraction=float(np.mean([float(r['temporal_valid_fraction']) for r in times])) if times else None))
        traces={};lrs={}
        for arm in ['A','U','B']:
            run=NEW/'runs'/args.trial/f'{label}_{arm}';complete=json.loads((run/'complete.json').read_text());assert complete['stage_updates']==5000 and complete['total_actual_updates']==52999
            for it in [49000,51000,53000]:assert json.loads((run/'validation'/f'{it}.json').read_text())['state']=='passed'
            files=sorted(run.glob('trace_*.jsonl'));assert len(files)==1,'Resume requires explicit discarded-update audit'
            trace=[json.loads(s) for s in files[0].read_text().splitlines()];assert [r['stage_update'] for r in trace]==list(range(1,5001))
            traces[arm]=[r['uid'] for r in trace];lrs[arm]=[r['lr'] for r in trace]
        assert traces['A']==traces['U']==traces['B'] and lrs['A']==lrs['U']==lrs['B']
        audit.append(dict(label=label,actual_sampling_identical=True,actual_lrs_identical=True,actual_extra_updates=5000,part_and_flow_denominators_identical=True))
    csvfile(stage/'paired_metrics.csv',globalrows);csvfile(stage/'part_metrics_summary.csv',localrows)
    spec=importlib.util.spec_from_file_location('frozen_video',BASE/'backend/src/video_io.py');video=importlib.util.module_from_spec(spec);spec.loader.exec_module(video);media=[]
    for label in ['N04','N11']:
        folder=stage/label;folder.mkdir();selection=json.loads((ROOT/'gallery'/label/'selection.json').read_text())['parts']
        sheets={name:Image.new('RGB',(1200,1448),'#282828') for name in ['rgb','alpha']}
        for name,sheet in sheets.items():ImageDraw.Draw(sheet).text((5,8),f'{label} {name}: frozen historical part crops, unmasked predictions',fill='white')
        for view in [0,8,16]:
            directories=[ROOT/'optimization/raw'/f'{label}_alpha',ROOT/'optimization/raw'/f'{label}_combined']+[NEW/'cache/raw'/args.trial/f'{label}_{arm}' for arm in ['A','U','B']]
            arrays=[]
            for directory in directories:
                assert (directory/'complete.json').exists()
                with np.load(directory/f'view{view:02d}.npz') as z:arrays.append((z['rgb'],z['alpha']))
            frames=len(arrays[0][0]);assert all(len(r)==len(a)==frames for r,a in arrays)
            writer=video.MP4Writer(str(folder/f'view{view:02d}.mp4'),25,2112,672)
            try:
                for frame in range(frames):
                    rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
                    images=[rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])]+[np.clip(r[frame]+.5*(1-a[frame,...,None]),0,1) for r,a in arrays]
                    alphas=[rgba[...,3]]+[a[frame] for _,a in arrays]
                    canvas=Image.new('RGB',(2112,672),'#282828');canvas.paste(Image.fromarray(np.rint(np.concatenate(images,1)*255).astype(np.uint8)),(0,32));ink=ImageDraw.Draw(canvas)
                    for i,name in enumerate(NAMES):ink.text((352*i+4,9),f'{name} | {label} v{view:02d} f{frame}',fill='white')
                    writer.append_data(np.asarray(canvas))
                    if frame in [0,(frames-1)//2,frames-1]:canvas.save(folder/f'view{view:02d}_frame{frame:03d}.jpg',quality=95)
                    for row,chosen in enumerate(selection):
                        if chosen.get('view')!=view or chosen.get('representative_frame')!=frame:continue
                        x0,y0,x1,y1=chosen['crop_xyxy'];ybase=32+236*row
                        for mode,values in [('rgb',images),('alpha',alphas)]:
                            sheet=sheets[mode];ink=ImageDraw.Draw(sheet)
                            for col,(name,picture) in enumerate(zip(NAMES,values)):
                                im=Image.fromarray(np.rint(picture[y0:y1,x0:x1]*255).astype(np.uint8)).convert('RGB');im=ImageOps.contain(im,(200,200));sheet.paste(im,(200*col+(200-im.width)//2,ybase+32+(200-im.height)//2))
                                ink.text((200*col+3,ybase+2),f'{name} {chosen["part"]}',fill='white');ink.text((200*col+3,ybase+16),f'v{view:02d} f{frame}',fill='white')
            finally:writer.close()
            path=folder/f'view{view:02d}.mp4';cap=cv2.VideoCapture(str(path));count=0
            while True:
                ok,image=cap.read()
                if not ok:break
                assert image.shape==(672,2112,3);count+=1
            cap.release();assert count==frames;media.append(dict(path=str(path.relative_to(stage)),frames=count,sha256=sha(path)))
        for mode,sheet in sheets.items():sheet.save(folder/f'parts_{mode}.jpg',quality=95)
    html=['<!doctype html><meta charset="utf-8"><title>Grid4D 边界监督配对对照</title><style>body{font:16px sans-serif;margin:24px;background:#eee}video,img{max-width:100%;height:auto}section{background:white;padding:18px;margin:20px 0}table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:7px}</style><h1>Grid4D：共同 alpha 起点的三组配对实验</h1><p>六列：生成参考 → 共同 alpha 起点 → 历史 combined → A 保持 alpha → U 全局加强 → B 边界加强。每组新增5000次更新；这是生成开发集评价，不是真实 Waymo 新视角精度。N11单列，不并入主七人均值。</p><p><a href="../FINDINGS.md">结论与限制（报告完成后可用）</a> · <a href="paired_metrics.csv">完整指标CSV</a> · <a href="part_metrics_summary.csv">部位与时间指标</a></p><table><tr><th>人物</th><th>组别</th>'+''.join('<th>'+x+'</th>' for x in FIELDS)+'</tr>']
    html+=['<tr><td>'+r['label']+'</td><td>'+r['method']+'</td>'+''.join(f'<td>{r[k]:.5f}</td>' for k in FIELDS)+'</tr>' for r in globalrows];html.append('</table>')
    for label in ['N04','N11']:
        html.append(f'<section><h2>{label}</h2><h3>固定部位 RGB 与直接渲染 alpha</h3><img loading="lazy" src="{label}/parts_rgb.jpg"><img loading="lazy" src="{label}/parts_alpha.jpg">')
        for view in [0,8,16]:html.append(f'<h3>视角{view:02d}全时长对照</h3><video controls loop preload="metadata" src="{label}/view{view:02d}.mp4"></video>')
        html.append('</section>')
    (stage/'index.html').write_text('\n'.join(html));save(stage/'validation.json',dict(state='passed',at=now(),paired=audit,videos=media,columns=NAMES,visual_quality_review='pending',reference='Frozen generated development set'))
    stage.rename(final)
    if args.trial=='primary':
        (NEW/'paired_metrics.csv').symlink_to('comparison/paired_metrics.csv')
    print('Published',final,'six complete videos',flush=True)


if __name__=='__main__':main()
