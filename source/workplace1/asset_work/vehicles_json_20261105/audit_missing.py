"""Diagnose every original no-source target without changing the live queue."""
import collections,itertools,json
from pathlib import Path
import ijson,numpy as np
from PIL import Image,ImageDraw
from prepare import ROOT,DATA,INDEX,read,write,rot

rows=[r for r in read(ROOT/'inventory.json')['assets'] if not r['candidates']]
wanted={r['instance_token'] for r in rows};anns=collections.defaultdict(list)
with (DATA/'v1.0-trainval/sample_annotation.json').open('rb') as f:
 for a in ijson.items(f,'item',use_float=True):
  if a['instance_token'] in wanted:anns[a['instance_token']].append(a)
signs=np.array(list(itertools.product([-1,1],repeat=3)));results=[]
sheet=Image.new('RGB',(1200,240*len(rows)),'white');draw=ImageDraw.Draw(sheet)
for n,r in enumerate(rows):
 ix=read(INDEX/(r['scene_name']+'.json'));frames=collections.defaultdict(list)
 for f in ix['frames']:frames[f['sample_token']].append(f)
 reasons=collections.Counter();views=[]
 for a in anns[r['instance_token']]:
  size=[a['size'][1],a['size'][0],a['size'][2]];world=(signs*np.array(size)/2)@rot(a['rotation']).T+np.array(a['translation'])
  for f in frames.get(a['sample_token'],[]):
   tf=np.linalg.inv(f['camera_to_world']);xyz=world@tf[:3,:3].T+tf[:3,3]
   if xyz[:,2].min()<=.1:reasons['behind_or_crosses_camera']+=1;continue
   uv=xyz@np.array(f['camera_intrinsic']).T;uv=uv[:,:2]/uv[:,2:];low=uv.min(0);high=uv.max(0)
   clow=np.maximum(low,0);chigh=np.minimum(high,[1600,900]);wh=chigh-clow
   if (wh<1).any():reasons['outside_image']+=1;continue
   failures=[]
   if (low<3).any() or (high>[1597,897]).any():failures.append('projected_box_touches_edge')
   if wh[0]<40 or wh[1]<30:failures.append('small_target')
   if int(a['visibility_token'])<3:failures.append('low_native_visibility')
   if not Path(f['path']).exists():failures.append('missing_image')
   reasons.update(failures)
   if 'missing_image' in failures:continue
   frame={k:v for k,v in f.items() if k!='motion_paths'}
   views.append(dict(frame=frame,annotation_token=a['token'],bbox_xyxy=[*clow.tolist(),*chigh.tolist()],unclipped_bbox=[*low.tolist(),*high.tolist()],size_lwh_m=size,visibility_token=a['visibility_token'],failed_filters=failures,score=float(wh.prod())*(.2 if 'projected_box_touches_edge' in failures else 1)))
 views.sort(key=lambda v:-v['score']);selected=[]
 for v in views:
  if any(v['frame']['camera_name']==o['frame']['camera_name'] and abs(v['frame']['timestamp_us']-o['frame']['timestamp_us'])<1000000 for o in selected):continue
  selected.append(v)
  if len(selected)==3:break
 out=ROOT/'missing_source_audit'/r['obj_id'];out.mkdir(parents=True,exist_ok=True)
 for j,v in enumerate(selected):
  im=Image.open(v['frame']['path']).convert('RGB');b=v['bbox_xyxy'];box=[max(0,int(b[0])-8),max(0,int(b[1])-8),min(1600,int(b[2])+9),min(900,int(b[3])+9)];im=im.crop(box);im.save(out/f'candidate_{j}.jpg');im.thumbnail((390,180));sheet.paste(im,(400*j+(390-im.width)//2,240*n+45));draw.text((400*j+3,240*n+3),f'{n}: {r["scene_name"]} {r["instance_token"][:8]} / v{v["visibility_token"]}',fill='black');draw.text((400*j+3,240*n+20),','.join(v['failed_filters']),fill='black')
 result=dict(obj_id=r['obj_id'],scene_name=r['scene_name'],instance_token=r['instance_token'],annotation_count=len(anns[r['instance_token']]),reasons=dict(reasons),candidates=selected);write(out/'audit.json',result);results.append(result)
sheet.save(ROOT/'missing_source_audit/contact_sheet.jpg')
write(ROOT/'missing_source_audit/summary.json',dict(targets=results,diagnostic_only=True))
for r in results:print(r['scene_name'],r['instance_token'][:8],r['reasons'],flush=True)
