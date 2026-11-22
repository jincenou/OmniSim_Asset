"""Render the first 224 frozen outputs as 28 labeled 4x2 Gaussian orbit videos."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

B = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('ASSET_PREVIEW_OUTPUT', str(B / 'preview224')))
P = Path('/mnt/ojc/pre_flux_pipeline')
PYTHON = '/home/wmj/miniconda3/envs/sim/bin/python'
GPUS = [int(v) for v in os.environ.get('ASSET_PREVIEW_GPUS', '0,2,3,4,5,6,7').split(',')]
sys.path.insert(0, str(P))
from asset_extraction.io_utils import read_json, write_json


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def register():
    if os.environ.get('ASSET_PREVIEW_MANIFEST'):
        document = read_json(os.environ['ASSET_PREVIEW_MANIFEST'])
        if not document['groups'] or any(not 0 < len(g) <= 8 for g in document['groups']):
            raise ValueError('Preview requires nonempty groups of at most eight assets')
        write_json(OUT / 'manifest.json', document)
        return document
    rows = []
    for path in sorted((B / 'chunks').glob('*/generation/generation.json')):
        generation = read_json(path)
        audit_path = path.parents[1] / 'curation/curation.json'
        audited = read_json(audit_path)['assets'] if audit_path.exists() else {}
        for asset in generation['assets']:
            if asset['status'] != 'generated':
                continue
            result = audited.get(asset['obj_id'], {})
            status = result.get('status', 'pending')
            if status == 'failed':
                status = 'audit_error'
            rows.append(dict(index=len(rows)+1, obj_id=asset['obj_id'], category=asset['category'],
                scene_id=asset['scene_id'], route=asset['route'], real_count=len(asset['real_images']),
                raw_path=asset['raw_path'], raw_sha256=asset['raw_sha256'],
                source_image=asset['image_path'], source_sha256=asset['image_sha256'],
                audit_status=status, audit_error=result.get('error'),
                defects=result.get('entry', {}).get('defects', [])))
    if len(rows) != 224:
        raise ValueError(f'Expected exactly the requested 224 existing outputs, got {len(rows)}')
    document = dict(asset_count=224, groups=[rows[start:start+8] for start in range(0,224,8)],
        order='frozen chunk then item index; no quality-based selection',
        layout='4 columns x 2 rows', frames=128, fps=16, duration_seconds=8,
        resolution=[2048,1152], route_counts=dict(Counter(r['route'] for r in rows)),
        audit_counts=dict(Counter(r['audit_status'] for r in rows)),
        scope='Raw generated Gaussians, preview-only axis alignment and grounding; not a curated asset bank.',
        camera='360-degree orbit, elevation12deg, robust bounds framing. No metric scale claimed.',
        source_inset='Original segmented source crop, not a generated image.')
    target = OUT / 'manifest.json'
    if target.exists():
        previous = read_json(target)
        old = [(r['obj_id'],r['raw_sha256']) for group in previous['groups'] for r in group]
        new = [(r['obj_id'],r['raw_sha256']) for group in document['groups'] for r in group]
        if old != new:
            raise ValueError('The fixed 224-output preview cohort changed')
        return previous
    write_json(target,document)
    return document


def worker(group_ids):
    import tempfile
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from gsplat import rasterization
    from asset_extraction.export import canonicalize_raw
    sys.path.insert(0, '/mnt/wmj/workspace4/pipeline')
    from curate.core import load_rigid
    document = read_json(OUT / 'manifest.json')
    font_path = '/home/wmj/miniconda3/envs/sim/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path,17)
    small = ImageFont.truetype(font_path,14)
    tile, header, frames = 512,64,document['frames']
    palette = dict(keep='#58d68d',reject='#ff8888',review='#ffd166',pending='#c5ced9',audit_error='#ffa94d')
    torch.set_num_threads(4)
    def tensor(value):
        return torch.as_tensor(value,device='cuda',dtype=torch.float32)
    for group_id in group_ids:
        group = document['groups'][group_id]
        name = f'grid_{group_id+1:02d}'
        video, evidence = OUT / f'{name}.mp4', OUT / f'{name}.json'
        if evidence.exists():
            record=read_json(evidence)
            if sha(video)!=record['video_sha256']:
                raise ValueError('Completed preview video changed')
            print(name,'verified',flush=True)
            continue
        started=time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f'.{name}_',dir=OUT) as temporary:
            assets=[]
            for row in group:
                if sha(row['raw_path'])!=row['raw_sha256'] or sha(row['source_image'])!=row['source_sha256']:
                    raise ValueError('Preview source changed')
                path=Path(temporary)/f'{row["index"]}.ply'
                if row.get('payload_is_canonical'):
                    path = Path(row['raw_path'])
                    canonical = row['canonical']
                else:
                    canonical=canonicalize_raw(row['raw_path'],path,route=row['route'],category=row['category'])
                gs=load_rigid(path)
                lo,hi=np.quantile(gs.xyz[0],[.005,.995],axis=0)
                center=(lo+hi)/2;radius=float(np.linalg.norm(hi-lo)/2)
                distance=radius*2.2/.78;focal=.5*tile*distance/radius*.78
                quats=tensor(gs.rot[0]);quats/=quats.norm(dim=-1,keepdim=True)
                assets.append(dict(tensors=[tensor(gs.xyz[0]),quats,tensor(gs.scale[0]),tensor(gs.opacity),tensor(gs.color)],
                    center=center,distance=distance,K=tensor([[focal,0,tile/2],[0,focal,tile/2],[0,0,1]])[None],
                    transform={k:v for k,v in canonical.items() if k!='canonical_path'},foreground=[]))
            base=Image.new('RGB',(tile*4,(tile+header)*2),'#202a36');draw=ImageDraw.Draw(base)
            for pos,row in enumerate(group):
                x,y=pos%4*tile,pos//4*(tile+header)
                label=f'{row["index"]:03d} | {row["category"]} | {row["route"]}'
                if row['route']=='H':label+=f' real={row["real_count"]}'
                draw.text((x+10,y+6),label,font=font,fill='white')
                draw.text((x+10,y+32),f'{row["obj_id"]}  QC: {row["audit_status"]}',font=small,fill=palette[row['audit_status']])
                source=ImageOps.contain(Image.open(row['source_image']).convert('RGB'),(64,52))
                base.paste(source,(x+440+(64-source.width)//2,y+4+(52-source.height)//2))
                draw.text((x+439,y+49),'SOURCE',font=small,fill='#adc1d5')
            temporary_video=OUT/f'.{name}.part.mp4'
            writer=imageio.get_writer(temporary_video,fps=document['fps'],codec='libx264',pixelformat='yuv420p',
                macro_block_size=16,ffmpeg_log_level='error',output_params=['-crf','22','-preset','fast','-threads','2','-movflags','+faststart'])
            try:
                for frame_index in range(frames):
                    azimuth=2*np.pi*frame_index/frames;el=np.deg2rad(12)
                    direction=np.array([np.cos(azimuth)*np.cos(el),np.sin(azimuth)*np.cos(el),np.sin(el)])
                    forward=-direction;right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
                    rotation=np.stack([right,np.cross(forward,right),forward])
                    canvas=base.copy()
                    for pos,asset in enumerate(assets):
                        view=np.eye(4);view[:3,:3]=rotation
                        view[:3,3]=-rotation@(asset['center']+asset['distance']*direction)
                        with torch.inference_mode():
                            rgb,alpha,_=rasterization(*asset['tensors'],tensor(view)[None],asset['K'],tile,tile,
                                backgrounds=torch.full((1,3),.78,device='cuda'),packed=False)
                        if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
                            raise ValueError('Nonfinite Gaussian render')
                        asset['foreground'].append(int((alpha[0,:,:,0]>.01).sum().item()))
                        pixels=(rgb[0,:,:,:3].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
                        canvas.paste(Image.fromarray(pixels),(pos%4*tile,pos//4*(tile+header)+header))
                    writer.append_data(np.asarray(canvas))
                    if frame_index in [0,32,64,96]:
                        canvas.save(OUT/f'{name}_angle{frame_index*360//frames:03d}.jpg',quality=90)
            finally:
                writer.close()
            temporary_video.replace(video)
            write_json(evidence,dict(group=group_id+1,video=str(video),video_sha256=sha(video),
                frames=frames,fps=document['fps'],seconds=time.monotonic()-started,
                assets=[dict(**row,preview_transform=asset['transform'],
                    foreground_min=min(asset['foreground']),foreground_max=max(asset['foreground']),
                    all_128_nonempty=all(n>0 for n in asset['foreground'])) for row,asset in zip(group,assets)]))
            del assets,gs,rgb,alpha
            torch.cuda.empty_cache()
            print(name,'complete',round(time.monotonic()-started,1),'seconds',flush=True)


async def run():
    document=register()
    async def launch(gpu,groups):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                 PYTHONDONTWRITEBYTECODE='1',CPATH='/home/wmj/miniconda3/envs/splatad/include',
                 PATH=str(Path(PYTHON).parent)+':'+os.environ['PATH'],
                 TORCH_EXTENSIONS_DIR='/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions',MAX_JOBS='2')
        with (OUT/f'worker_gpu{gpu}.log').open('a') as log:
            process=await asyncio.create_subprocess_exec(PYTHON,str(Path(__file__).resolve()),'--groups',*map(str,groups),
                cwd=P,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
            try:
                code=await process.wait()
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
            if code:raise RuntimeError(f'Preview GPU {gpu} exited {code}; inspect its log')
    count = len(document['groups'])
    await asyncio.gather(*(launch(gpu,list(range(slot,count,len(GPUS))))
                          for slot,gpu in enumerate(GPUS) if slot < count))
    sections=[]
    for i,group in enumerate(document['groups'],1):
        record=read_json(OUT/f'grid_{i:02d}.json')
        if len(record['assets'])!=len(group) or sha(OUT/f'grid_{i:02d}.mp4')!=record['video_sha256']:
            raise ValueError('Incomplete preview grid')
        labels='、'.join(f'{r["index"]:03d} {html.escape(r["category"])}' for r in group)
        sections.append(f'<article><h2>第 {i:02d} 组 · {group[0]["index"]:03d}–{group[-1]["index"]:03d}</h2>'
            f'<video controls loop playsinline preload="none" poster="grid_{i:02d}_angle000.jpg" src="grid_{i:02d}.mp4"></video>'
            f'<p>{labels}</p><a href="grid_{i:02d}.mp4" download>下载本组视频</a></article>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page+='<title>224 个资产 · 28 组环绕视频</title><style>body{margin:24px auto;padding:0 20px;max-width:1400px;background:#111923;color:#e7eef7;font:16px/1.6 system-ui}video{width:100%;background:#202a36}article{margin:32px 0;padding:16px;background:#1c2735}a{color:#8ac6ff}h1{font-size:28px}</style>'
    page+='<h1>224 个原始高斯资产 · 28 组 4×2 环绕视频</h1><p>全部来自 nuScenes scene-0001，按生成顺序展示，每段 8 秒。灰底是真实高斯渲染，SOURCE 小图为原始分割裁图。H 支持 0–4 张真实图；B0 使用原 TRELLIS。各对象按自身尺寸取景，画面大小不表示米制尺寸。</p>'
    page+='<p>QC 是制作视频时的审核快照：keep=通过，reject=未通过，pending=尚未审核，audit_error=审核请求出错。所有结果均保留展示，不做按质量挑选。这里只统一预览坐标及接地，未强行规定每个物体的真实正面；不代表已入库或已验证场景插入。</p>'
    if document.get('scope') == 'curated_scene_pilots':
        title = html.escape(document.get('title', f'两场景最终保留资产 · {document["asset_count"]} 个'))
        description = html.escape(document.get('description', '按数据集、类别展示筛选通过的规范化高斯资产。'))
        page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        page += f'<title>{title}</title><style>body{{margin:24px auto;padding:0 20px;max-width:1400px;background:#111923;color:#e7eef7;font:16px/1.6 system-ui}}video{{width:100%}}article{{margin:32px 0}}a{{color:#8ac6ff}}</style>'
        page += f'<h1>{title}</h1><p>{description}每组最多 8 个，4×2 排列，8 秒环绕。原始分割图位于 SOURCE 小窗。物体分别取景，显示大小不表示真实尺寸。</p><p><a href="../START_HERE.html">返回完整筛选统计与未通过结果</a></p>'
    (OUT/'START_HERE.html').write_text(page+''.join(sections)+'</html>')
    write_json(OUT/'status.json',dict(status='completed',videos=count,assets=document['asset_count'],frames_per_video=document['frames'],
        total_video_bytes=sum(p.stat().st_size for p in OUT.glob('grid_*.mp4'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--groups',type=int,nargs='+');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.groups is not None:worker(args.groups)
    else:asyncio.run(run())
