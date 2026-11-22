"""Rank local has-rider instances by native projected extent; no model quality claims."""
import json, itertools
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from PIL import Image,ImageDraw
root=Path('/mnt/public_data/nuScenes'); out=Path(__file__).resolve().parents[1]/'sources/nuscenes';out.mkdir(exist_ok=True)
def table(name):return json.loads((root/'v1.0-trainval'/f'{name}.json').read_text())
def index(name):return {x['token']:x for x in table(name)}
def rot(q):
 w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
cats=index('category');instances=index('instance');attrs=index('attribute');scenes=index('scene');samples=index('sample')
anns=[a for a in table('sample_annotation') if cats[instances[a['instance_token']]['category_token']]['name'] in ['vehicle.motorcycle','vehicle.bicycle'] and any(attrs[t]['name']=='cycle.with_rider' for t in a['attribute_tokens'])]
print('candidate identities',dict(Counter(cats[instances[t]['category_token']]['name'] for t in {a['instance_token'] for a in anns})),flush=True)
by_sample=defaultdict(list)
for a in anns:by_sample[a['sample_token']].append(a)
cal=index('calibrated_sensor');ego=index('ego_pose');sensors=index('sensor');rows=[]
for sd in table('sample_data'):
 if not sd['is_key_frame'] or sd['sample_token'] not in by_sample:continue
 c=cal[sd['calibrated_sensor_token']]; sen=sensors[c['sensor_token']]
 if sen['modality']!='camera':continue
 path=root/sd['filename']
 if not path.is_file():continue
 e=ego[sd['ego_pose_token']]; K=np.array(c['camera_intrinsic']); Re=rot(e['rotation']); Rc=rot(c['rotation'])
 for a in by_sample[sd['sample_token']]:
  if int(a['visibility_token'])<3:continue
  w,l,h=a['size'];corn=np.array(list(itertools.product([-l/2,l/2],[-w/2,w/2],[-h/2,h/2])))
  world=corn@rot(a['rotation']).T+np.array(a['translation'])
  cam=((world-np.array(e['translation']))@Re-np.array(c['translation']))@Rc
  if cam[:,2].min()<1:continue
  p=cam@K.T;p=p[:,:2]/p[:,2:];lo=p.min(0);hi=p.max(0)
  if (lo<8).any() or hi[0]>sd['width']-8 or hi[1]>sd['height']-8:continue
  wh=hi-lo
  if min(wh)<65 or max(wh)<145:continue
  rows.append(dict(instance=a['instance_token'],annotation=a['token'],sample=a['sample_token'],sample_data=sd['token'],scene=scenes[samples[a['sample_token']]['scene_token']]['name'],category=cats[instances[a['instance_token']]['category_token']]['name'].split('.')[-1],image_path=str(path),timestamp_us=sd['timestamp'],bbox= [*lo.tolist(),*hi.tolist()],size_wlh=a['size'],camera=sen['channel'],native_wh=wh.tolist(),score=float(np.prod(wh)),visibility=a['visibility_token']))
byid=defaultdict(list)
for r in rows:byid[r['instance']].append(r)
ranked=sorted(byid,key=lambda k:(byid[k][0]['category']=='motorcycle',max(r['score'] for r in byid[k])),reverse=True)
(out/'observations.json').write_text(json.dumps(rows,indent=2));(out/'ranked.json').write_text(json.dumps(ranked,indent=2))
for page in range(min(3,(len(ranked)+23)//24)):
 sheet=Image.new('RGB',(6*260,4*240),'#eeeeee');d=ImageDraw.Draw(sheet)
 for j,k in enumerate(ranked[page*24:(page+1)*24]):
  r=max(byid[k],key=lambda r:r['score']);im=Image.open(r['image_path']);x1,y1,x2,y2=r['bbox'];pad=10
  crop=im.crop((max(0,x1-pad),max(0,y1-pad),min(im.width,x2+pad),min(im.height,y2+pad)));crop.thumbnail((250,205));x=j%6*260;y=j//6*240
  sheet.paste(crop,(x+(260-crop.width)//2,y));d.text((x+5,y+207),f"{page*24+j:02d} {r['category']} {k[:8]}\n{r['scene']} {int(x2-x1)}x{int(y2-y1)}",fill='black')
 sheet.save(out/f'candidates_{page}.jpg')
print('local eligible identities',len(ranked),'observations',len(rows),flush=True)
