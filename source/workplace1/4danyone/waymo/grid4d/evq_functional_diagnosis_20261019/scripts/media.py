from pathlib import Path
import sys,csv,json,math,argparse
import numpy as np,cv2
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parents[1];ROOT=D.parent;OLD=ROOT/'early_visual_query_20261019';BASE=ROOT.parent/'random8/r20261015_113332';sys.path.insert(0,str(BASE/'backend/src'))
from video_io import MP4Writer
from summarize import write

def ref(label,v,f):
 a=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{v:02d}'/f'{f:06d}.png'),np.float32)/255;return a[...,:3]*a[...,3:]+.5*(1-a[...,3:]),a[...,3]
def canvas(images,names,f,box=None):
 if box is not None:
  x0,y0,x1,y1=box;images=[x[y0:y1,x0:x1] for x in images]
 h,w=images[0].shape[:2];W=w*len(images);H=h+26;W+=W%2;H+=H%2
 out=Image.new('RGB',(W,H),'#303030');dr=ImageDraw.Draw(out)
 for i,(x,name) in enumerate(zip(images,names)):
  out.paste(Image.fromarray(np.rint(np.clip(x,0,1)*255).astype('uint8')),(i*w,26));dr.text((i*w+4,6),f'{name} f{f}',fill='white')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();key=a.key;label=key[:3];plan=json.loads((D/'PLAN_DIAGNOSTIC.json').read_text());sel=plan['selections'][label];out=D/'comparison'/key;out.mkdir(exist_ok=True);cache=D/'cache'/key;win=sel['window'];v=win['view'];cv2.setNumThreads(1);names=['REF','M00 base','M10 pos','M01 color','M11 full'];files=[]
 with np.load(OLD/'data/expanded_roi'/label/f'view{v:02d}.npz') as z:box=z['boxes'][0]
 contacts=[];frames=[]
 for f in range(win['start'],win['end']+1):
  with np.load(cache/f'view{v:02d}_{f:03d}.npz') as z:pred=z['pred']
  im=canvas([ref(label,v,f)[0]]+list(pred),names,f,box);frames.append(im)
 w,h=frames[0].size;path=out/'fixed_window.mp4'
 with MP4Writer(str(path),25,w,h) as writer:
  for im in frames:writer.append_data(np.asarray(im))
 cap=cv2.VideoCapture(str(path));count=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape==(h,w,3);count+=1
 cap.release();assert count==25;files.append(dict(path=str(path.relative_to(D)),frames=count,fully_decoded=True))
 for st in range(0,25,5):
  page=Image.new('RGB',(w,h*5),'#303030')
  for j in range(5):page.paste(frames[st+j],(0,j*h))
  page.save(out/f'window_{win["start"]+st:03d}_{win["start"]+st+4:03d}.jpg',quality=95)
 page=Image.new('RGB',(w,h*3))
 for j,f in enumerate([0,12,24]):page.paste(frames[f],(0,j*h))
 page.save(out/'window_contact.jpg',quality=95)
 for v in [0,8,16]:
  imlist=[]
  for f in [0,sel['frames']//2,sel['frames']-1]:
   with np.load(cache/f'view{v:02d}_{f:03d}.npz') as z:pred=z['pred'];alpha=z['alpha']
   imlist.append(canvas([ref(label,v,f)[0]]+list(pred),names,f))
   if f==sel['frames']-1:imlist[-1].save(out/f'last_v{v:02d}.jpg',quality=95)
  page=Image.new('RGB',(imlist[0].width,imlist[0].height*3))
  for j,im in enumerate(imlist):page.paste(im,(0,j*im.height))
  page.save(out/f'full_v{v:02d}.jpg',quality=95)
 # Signed alpha differences are a separate coverage visualization, not output repair.
 rows=[];representatives=[win['start'],(win['start']+win['end'])//2,win['end']]
 for v,f in sel['observations']:
  with np.load(cache/f'view{v:02d}_{f:03d}.npz') as z:alpha=z['alpha']
  with np.load(ROOT/'audit/roi'/label/f'view{v:02d}.npz') as z:lab=z['labels'][f]
  with np.load(OLD/'data/expanded_roi'/label/f'view{v:02d}.npz') as z:m=z['masks'][f,0]
  target,ra=ref(label,v,f);delta=alpha[3]-alpha[2]
  for name,mask in [('reference_foreground',ra>.5),('reference_background',ra<=.5),('torso_core',lab==2),('target_expanded',m)]:
   rows.append(dict(key=key,view=v,frame=f,region=name,pixels=int(mask.sum()),alpha_increase=float(np.maximum(delta,0)[mask].sum()),alpha_decrease=float(np.maximum(-delta,0)[mask].sum()),direction='M11 minus M01; direct position effect with color fixed',occlusion_layer='unknown;2D masks do not identify material occlusion layers'))
  if v==win['view'] and f in representatives:
   signed=np.ones((640,352,3),np.float32)*.5;signed[...,0]+=.5*np.maximum(delta,0);signed[...,2]+=.5*np.maximum(-delta,0);signed[...,1]-=.5*abs(delta)
   im=canvas([target,np.repeat(alpha[2,...,None],3,2),np.repeat(alpha[3,...,None],3,2),signed],['REF','alpha M01','alpha M11','red+ blue-'],f,box);im.save(out/f'alpha_f{f:03d}.jpg',quality=95)
 write(D/'results'/key/'alpha_transfer.csv',rows);(out/'media.json').write_text(json.dumps(dict(key=key,files=files,manual_review='pending; decoding only'),indent=2));print(key,'media complete',flush=True)
if __name__=='__main__':main()
