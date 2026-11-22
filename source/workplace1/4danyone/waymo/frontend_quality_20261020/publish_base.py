"""Publish complete Base comparisons only after every target video decodes."""
import argparse,json,shutil,sys
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'scene20/r20261019/scripts'))
from deliver import encode

def main():
 parser=argparse.ArgumentParser();parser.add_argument('case');args=parser.parse_args()
 old=ROOT.parent/'scene20/r20261019/scenes'/args.case;work=ROOT/'base_ablation'/old.name
 out=work/'generation/outputs/base';meta=json.loads((out/'metadata.json').read_text())
 count=json.loads((old/'generation.json').read_text())['output']['frames_per_video']
 assert meta['output']['frames_per_video']==count and meta['generation']['inference_steps']==24
 assert meta['model'].get('turbo_lora') is None,meta['model']
 validation=[];selected={};views=[0,4,8,12,16,20]
 for view in range(24):
  path=out/'videos'/f'{view:02d}.mp4';images=[];n=0
  with av.open(str(path)) as c:
   c.streams.video[0].thread_count=1
   for i,frame in enumerate(c.decode(video=0)):
    assert (frame.width,frame.height)==(704,1280)
    assert abs(float(frame.pts*frame.time_base)-i/25)<1e-8
    if view in views:images.append(frame.to_image().resize((264,480)))
    n+=1
  assert n==count,(view,n,count)
  validation.append({'view':view,'decoded_frames':n})
  if view in views:selected[view]=images
 cameras=json.loads((out/'cameras.json').read_text());reference=json.loads((old/'cameras.json').read_text())
 for a,b in zip(cameras['cameras'],reference['cameras'],strict=True):
  np.testing.assert_allclose(a['K'],b['K'],rtol=0,atol=1e-6)
  np.testing.assert_allclose(a['camera_to_world'],b['camera_to_world'],rtol=0,atol=1e-6)
 with av.open(str(old/'preview.mp4')) as c:
  c.streams.video[0].thread_count=1;baseline=[f.to_image() for f in c.decode(video=0)]
 assert len(baseline)==count
 preview=[];comparison=[]
 for t in range(count):
  im=Image.new('RGB',(792,1000),'#222222');draw=ImageDraw.Draw(im)
  for j,v in enumerate(views):
   x=j%3*264;y=j//3*500;im.paste(selected[v][t],(x,y+20));draw.text((x+4,y+3),f'Base view{v:02d} frame{t}',fill='white')
  preview.append(im)
  panel=Image.new('RGB',(1584,1024),'#222222');panel.paste(baseline[t],(0,24));panel.paste(im,(792,24))
  draw=ImageDraw.Draw(panel);draw.text((4,5),'Baseline Turbo4',fill='white');draw.text((796,5),'Base24: same original input and motion',fill='white');comparison.append(panel)
 # No public comparison is written until all 24 videos and cameras pass above.
 result=work/'result'
 if result.exists():raise FileExistsError(result)
 shutil.move(str(out),str(result));shutil.copy2(work/'generation/preflight/root_motion.npz',result/'root_motion.npz')
 cameras['motion_world_to_local_world_file']='root_motion.npz';cameras['root_motion']={'file':'root_motion.npz','scope':'Reused baseline estimated root motion, not Waymo world registration'}
 (result/'cameras.json').write_text(json.dumps(cameras,indent=2)+'\n')
 encode(result/'preview.mp4',preview,25);encode(result/'comparison.mp4',comparison,25)
 contact=Image.new('RGB',(1584,1024*3))
 for i,t in enumerate([0,count//2,count-1]):contact.paste(comparison[t],(0,1024*i))
 contact.save(result/'comparison.jpg',quality=93)
 for name in ['preview.mp4','comparison.mp4']:
  with av.open(str(result/name)) as c:assert sum(1 for _ in c.decode(video=0))==count
 (result/'validation.json').write_text(json.dumps({'state':'all24_decoded_and_comparison_published','videos':validation,'baseline':str(old),
  'appearance_or_motion_quality_approved':False,'visual_review':'pending','scope':'Same original input and motion; model schedule change only, crop reproduction limitation retained in pairing report.'},indent=2)+'\n')
 print(json.dumps({'sample':old.name,'result':str(result),'state':'published_needs_quality_review'}))
if __name__=='__main__':main()
