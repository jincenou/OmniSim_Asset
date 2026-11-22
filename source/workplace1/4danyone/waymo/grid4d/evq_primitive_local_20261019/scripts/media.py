"""Publish only complete four-arm terminal comparisons; fixed views and windows."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;BASE=ROOT.parent/'random8/r20261015_113332';sys.path.insert(0,str(BASE/'backend/src'))
from video_io import MP4Writer
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rgb(raw,alpha):return np.clip(raw+.5*(1-alpha[...,None]),0,1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);a=ap.parse_args();label=a.label;cv2.setNumThreads(2);plan=json.loads((NEW/'PLAN.json').read_text());inp=plan['inputs'][label];frames=inp['frame_count'];out=NEW/'comparison'/label
    for arm in plan['arms']:
        assert (NEW/'results'/f'{label}_{arm}_32999/parts.json').exists();assert json.loads((NEW/'assets'/f'{label}_{arm}/replay_validation.json').read_text())['passed']
    out.mkdir(parents=True,exist_ok=False);files=[];old=inp['old_references']
    paths={arm:NEW/'cache/raw'/f'{label}_{arm}_32999' for arm in plan['arms']};paths.update(A=Path(old['A']['raw']),R1=ROOT/'early_visual_query_20261019/cache/raw'/f'{label}_R1_32999')
    for view in [0,8,16]:
        arrays={}
        for name,path in paths.items():
            with np.load(path/f'view{view:02d}.npz') as z:arrays[name]=(z['rgb'],z['alpha']);assert len(z['alpha'])==frames
        schemes={'ablation':['reference','G0','G1','G2'],'history':['reference','R1','A','G2']}
        for tag,names in schemes.items():
            p=out/f'{tag}_v{view:02d}.mp4';contact=Image.new('RGB',(352*len(names),672*3),'#282828')
            with MP4Writer(str(p),25,352*len(names),672) as writer:
                for frame in range(frames):
                    rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255;ref=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);images=[ref if n=='reference' else rgb(arrays[n][0][frame],arrays[n][1][frame]) for n in names]
                    strip=np.rint(np.concatenate(images,1)*255).astype(np.uint8);canvas=Image.new('RGB',(352*len(names),672),'#282828');canvas.paste(Image.fromarray(strip),(0,32));ink=ImageDraw.Draw(canvas)
                    for i,n in enumerate(names):ink.text((352*i+5,9),f'{n} | {label} v{view:02d} f{frame}',fill='white')
                    writer.append_data(np.asarray(canvas))
                    if frame in [0,(frames-1)//2,frames-1]:contact.paste(canvas,(0,[0,(frames-1)//2,frames-1].index(frame)*672))
            contact.save(out/f'{tag}_v{view:02d}_contact.jpg',quality=95)
            cap=cv2.VideoCapture(str(p));count=0
            while True:
                ok,im=cap.read()
                if not ok:break
                assert im.shape==(672,352*len(names),3);count+=1
            cap.release();assert count==frames;files.append(dict(path=str(p.relative_to(NEW)),frames=count,sha256=sha(p),fully_decoded=True,columns=names))
        # Reference-defined supplementary visual crops, never metric ROI changes.
        times=[frames-1] if label=='N11' else [0,(frames-1)//2,frames-1]
        details=[];boxes_record=[]
        for f in times:
            rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{f:06d}.png'),np.float32)/255
            ref=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);ys,xs=np.where(rgba[...,3]>.5)
            assert len(xs)>0
            x0=max(0,int(xs.min())-12);x1=min(352,int(xs.max())+13);top=int(ys.min());bottom=min(640,int(ys.max())+13);height=bottom-top
            y0=int(top+(.65 if label=='N11' else .15)*height);y1=bottom if label=='N11' else int(top+.65*height)
            boxes_record.append(dict(frame=f,box=[x0,y0,x1,y1],reference_only=True))
            row=Image.new('RGB',(280*4,330),'#282828');draw=ImageDraw.Draw(row)
            for col,name in enumerate(schemes['ablation']):
                a=ref if name=='reference' else rgb(arrays[name][0][f],arrays[name][1][f])
                crop=Image.fromarray(np.rint(a[y0:y1,x0:x1]*255).astype(np.uint8));crop.thumbnail((280,300));row.paste(crop,(col*280+(280-crop.width)//2,30));draw.text((col*280+4,8),f'{name} f{f}',fill='white')
            details.append(row)
        detail=Image.new('RGB',(1120,330*len(details)))
        for j,row in enumerate(details):detail.paste(row,(0,j*330))
        detail.save(out/f'last_feet_v{view:02d}.jpg' if label=='N11' else out/f'torso_v{view:02d}.jpg',quality=95)
        (out/f'visual_crop_v{view:02d}.json').write_text(json.dumps(dict(scope='reference-body-fraction visual crop only;not semantic foot truth',boxes=boxes_record),indent=2))
        if label in ['N06','N11']:
            bad=plan['evaluation']['fixed_bad_windows'][label]
            if view==bad['view']:
                with np.load(NEW/'data/expanded_roi'/label/f'view{view:02d}.npz') as z:box=z['boxes'][0]
                x0,y0,x1,y1=map(int,box);sampleframes=[bad['frames'][0],sum(bad['frames'])//2,bad['frames'][1]];cw=x1-x0;ch=y1-y0;canvas=Image.new('RGB',(cw*4,(ch+28)*3),'#282828');ink=ImageDraw.Draw(canvas)
                for row,f in enumerate(sampleframes):
                    rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{f:06d}.png'),np.float32)/255;ref=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])
                    for col,n in enumerate(schemes['ablation']):
                        im=ref if n=='reference' else rgb(arrays[n][0][f],arrays[n][1][f]);canvas.paste(Image.fromarray(np.rint(im[y0:y1,x0:x1]*255).astype(np.uint8)),(cw*col,(ch+28)*row+28));ink.text((cw*col+2,(ch+28)*row+5),f'{n} f{f}',fill='white')
                canvas.save(out/'fixed_bad_window_left.jpg',quality=95)
                audit=NEW/'research/visual_windows'/label;audit.mkdir(parents=True,exist_ok=False)
                # Five pages jointly cover every frame of the predeclared25-slot
                # window, separate from the compact three-frame presentation.
                for start in range(bad['frames'][0],bad['frames'][1]+1,5):
                    times=list(range(start,min(start+5,bad['frames'][1]+1)));page=Image.new('RGB',(cw*4,(ch+28)*len(times)),'#282828');draw=ImageDraw.Draw(page)
                    for row,f in enumerate(times):
                        rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{f:06d}.png'),np.float32)/255;ref=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])
                        for col,n in enumerate(schemes['ablation']):
                            im=ref if n=='reference' else rgb(arrays[n][0][f],arrays[n][1][f]);page.paste(Image.fromarray(np.rint(im[y0:y1,x0:x1]*255).astype(np.uint8)),(cw*col,(ch+28)*row+28));draw.text((cw*col+2,(ch+28)*row+5),f'{n} f{f}',fill='white')
                    page.save(audit/f'frames_{times[0]:03d}_{times[-1]:03d}.jpg',quality=95)
    (out/'complete.json').write_text(json.dumps(dict(label=label,all_three_arms_complete=True,files=files,visual_inspection='pending;decode is not manual motion-quality inspection'),indent=2)+'\n');print(label,'complete comparisons published',flush=True)
if __name__=='__main__':main()
