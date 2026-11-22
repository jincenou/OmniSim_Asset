"""Publish only complete comparison families; historical crops and full frame decode."""
import asyncio,importlib.util,json,hashlib,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageOps
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;OLD=ROOT/'optimization_boundary_20261018';BASE=ROOT.parent/'random8/r20261015_113332'
spec=importlib.util.spec_from_file_location('video_io',BASE/'backend/src/video_io.py');video=importlib.util.module_from_spec(spec);spec.loader.exec_module(video);cv2.setNumThreads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def families():
    jobs=[]
    for label in ['N01','N07']:
        jobs.append(dict(name=label+'_A_fusion',label=label,columns=[('HUST grid61',ROOT/'tmp/baseline_raw'/label/'4dgs_grid61')]+[(name,NEW/'cache/raw'/f'{label}_{stage}_{epoch}') for name,stage,epoch in [('Native 32999','33k',33000),('Alpha 47999','alpha',48000),('A 52999','A',53000),('Fused 32999','fused',33000)]],note='Same frozen input and seed; independent native trajectories are not bitwise identical. Fusion15k pre-alpha snapshots retained.'))
    for label in ['N04','N11']:
        columns=[('Alpha 47999',ROOT/'optimization/raw'/f'{label}_alpha')]+[(arm,OLD/'cache/raw/primary'/f'{label}_{arm}') for arm in ['A','U','B']]
        if label=='N04':columns.append(('B-match',NEW/'cache/raw/N04_Bmatch_53000'))
        jobs.append(dict(name=label+'_boundary',label=label,columns=columns,note='N11 B-match not triggered by frozen selection rule; no fabricated new column.' if label=='N11' else 'Only added boundary coefficient changed after actual Adam calibration.'))
    jobs.append(dict(name='N11_color',label='N11',columns=[('A original',OLD/'cache/raw/primary/N11_A'),('C1 SH Adam',NEW/'cache/raw/N11_C1_20'),('C2 block residual',NEW/'cache/raw/N11_C2_20')],note='Only SH colors updated; geometry,opacity,hash,deform,degree and point identities frozen.'))
    return jobs
def publish(job):
    started=time.monotonic();name,label=job['name'],job['label'];out=NEW/'tmp/publish'/name;out.mkdir(parents=True,exist_ok=False);columns=[x[0] for x in job['columns']];width=352*(len(columns)+1);names=['Generated reference']+columns
    original_selection=json.loads((ROOT/'gallery'/label/'selection.json').read_text())['parts'];selection=[s for s in original_selection if 'crop_xyxy' in s];unknown=[s['part'] for s in original_selection if 'crop_xyxy' not in s];sheets={m:Image.new('RGB',(200*len(names),64+236*len(selection)),'#282828') for m in ['rgb','alpha']};records=[];contacts=[]
    for view in [0,8,16]:
        arrays=[]
        for title,folder in job['columns']:
            assert (folder/'complete.json').exists()
            with np.load(folder/f'view{view:02d}.npz') as z:arrays.append((z['rgb'],z['alpha']))
        count=len(arrays[0][0]);assert all(len(rgb)==len(a)==count for rgb,a in arrays)
        contact=Image.new('RGB',(width,672*3),'#282828');frames=[0,(count-1)//2,count-1];writer=video.MP4Writer(str(out/f'view{view:02d}.mp4'),25,width,672)
        try:
            for frame in range(count):
                rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
                pictures=[rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])]+[np.clip(rgb[frame]+.5*(1-a[frame,...,None]),0,1) for rgb,a in arrays];alphas=[rgba[...,3]]+[a[frame] for rgb,a in arrays]
                canvas=Image.new('RGB',(width,672),'#282828');canvas.paste(Image.fromarray(np.rint(np.concatenate(pictures,1)*255).astype(np.uint8)),(0,32));ink=ImageDraw.Draw(canvas)
                for i,title in enumerate(names):ink.text((352*i+4,8),f'{title} | {label} v{view:02d} f{frame}',fill='white')
                writer.append_data(np.asarray(canvas))
                if frame in frames:contact.paste(canvas,(0,672*frames.index(frame)))
                for row,chosen in enumerate(selection):
                    if chosen.get('view')!=view or chosen.get('representative_frame')!=frame:continue
                    x0,y0,x1,y1=chosen['crop_xyxy'];ybase=32+236*row
                    for mode,values in [('rgb',pictures),('alpha',alphas)]:
                        sheet=sheets[mode];ink=ImageDraw.Draw(sheet)
                        for col,(title,picture) in enumerate(zip(names,values)):
                            patch=Image.fromarray(np.rint(picture[y0:y1,x0:x1]*255).astype(np.uint8)).convert('RGB');patch=ImageOps.contain(patch,(200,200));sheet.paste(patch,(200*col+(200-patch.width)//2,ybase+32+(200-patch.height)//2));ink.text((200*col+3,ybase+2),f'{title} {chosen["part"]}',fill='white');ink.text((200*col+3,ybase+16),f'v{view:02d} f{frame}',fill='white')
        finally:writer.close()
        path=out/f'view{view:02d}.mp4';cap=cv2.VideoCapture(str(path));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(672,width,3);decoded+=1
        cap.release();assert decoded==count;records.append(dict(path=path.name,frames=count,width=width,height=672,fps=25,sha256=sha(path)));contact.save(out/f'view{view:02d}_contact.jpg',quality=94)
    for mode,sheet in sheets.items():
        ImageDraw.Draw(sheet).text((5,8),'Frozen historical crops; full unmasked predictions; estimated body cores',fill='white');ImageDraw.Draw(sheet).text((5,sheet.height-24),'Unknown cores: '+(', '.join(unknown) if unknown else 'none'),fill='white');sheet.save(out/f'parts_{mode}.jpg',quality=95)
    result=dict(state='complete_media_decoded_visual_review_pending',name=name,label=label,columns=names,note=job['note'],videos=records,seconds=time.monotonic()-started,selection_path=str(ROOT/'gallery'/label/'selection.json'),selection_sha256=sha(ROOT/'gallery'/label/'selection.json'))
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');out.rename(NEW/'comparison'/name);return result
async def main():
    jobs=families();pending=jobs.copy();tasks=[];sem=asyncio.Semaphore(2)
    async def bounded(job):
        async with sem:return await asyncio.to_thread(publish,job)
    while pending:
        for job in pending.copy():
            if all((p/'complete.json').exists() for name,p in job['columns']):tasks.append(asyncio.create_task(bounded(job)));pending.remove(job)
        if pending:await asyncio.sleep(20)
    results=await asyncio.gather(*tasks)
    html=['<!doctype html><meta charset="utf-8"><title>Grid4D A / 融合 / 边界 / 颜色对照</title><style>body{font:16px sans-serif;max-width:1600px;margin:auto;padding:24px;background:#eee}section{background:white;padding:20px;margin:20px 0}video,img{max-width:100%;height:auto}summary{cursor:pointer}</style><h1>Grid4D 固定终态对照</h1><p>生成参考上的开发评测，不是真实 Waymo 新视角真值。N01/N04/N07 主队列；N11 困难样本单列。所有视频已完整解码；视觉验收结论见报告。</p><p><a href="../REPORT.md">进度/报告</a> · <a href="../paired_metrics_partial.csv">当前指标 CSV</a></p>']
    for job in jobs:
        name=job['name'];html.append(f'<section><h2>{name}</h2><p>{job["note"]}</p><p>列顺序：Generated reference → '+ ' → '.join(x[0] for x in job['columns'])+'</p>')
        html.append(f'<h3>固定局部 RGB / alpha</h3><img loading="lazy" src="{name}/parts_rgb.jpg"><details><summary>查看 alpha</summary><img loading="lazy" src="{name}/parts_alpha.jpg"></details>')
        for view in [0,8,16]:html.append(f'<h3>视角 {view:02d}</h3><video controls loop preload="metadata" src="{name}/view{view:02d}.mp4"></video><details><summary>首/中/末帧</summary><img loading="lazy" src="{name}/view{view:02d}_contact.jpg"></details>')
        html.append('</section>')
    (NEW/'comparison/index.html').write_text('\n'.join(html));(NEW/'comparison/media_validation.json').write_text(json.dumps(dict(state='complete',families=results),indent=2)+'\n');print('All five complete comparison families published',flush=True)
if __name__=='__main__':asyncio.run(main())
