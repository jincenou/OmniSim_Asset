"""Rank new native tracks from existing audited inventories, using inputs only."""
from native_candidates import ROOT,frames
from pathlib import Path
from collections import defaultdict
import json,math,io
import numpy as np
from PIL import Image,ImageOps,ImageDraw

out=ROOT/'inputs/fresh_selection';out.mkdir(exist_ok=True)
exclude={'CcxBT9sbWSZEbE2GkV6A3Q','0al-qbyijVhonVpSNhGVhw','RplJ7o5oCzTl5lLLMPw9mA'}
all_tracks=[]
for scene in ['service_van','silver_sedan_03','white_sedan']:
 d=json.loads((ROOT/'inputs'/scene/'associated_inventory.json').read_text());tracks=defaultdict(list)
 for row in d['observations']:
  r=dict(row);lid=r.get('projected_laser_object_id');b=r['box_xyxy'];w,h=r['image_size']
  if not lid or lid in exclude or min(b[0],b[1],w-b[2],h-b[3])<5:continue
  if min(b[2]-b[0],b[3]-b[1])<75:continue
  box=r['projected_laser_box'];cam=np.array(r['camera_to_ego']).reshape(4,4)[:3,3]
  delta=cam[:2]-np.array(box[:2]);a=math.atan2(delta[1],delta[0])-box[6]
  r['view_azimuth_rad']=math.atan2(math.sin(a),math.cos(a));r['bbox_area']=(b[2]-b[0])*(b[3]-b[1]);tracks[lid].append(r)
 ranked=[]
 for lid,rows in tracks.items():
  if len(rows)<15:continue
  selected=[max(rows,key=lambda r:r['bbox_area'])]
  while len(selected)<10:
   avail=[r for r in rows if all(abs(r['frame_index']-s['frame_index'])>=5 for s in selected)]
   if not avail:break
   def distance(r):return min(abs(math.atan2(math.sin(r['view_azimuth_rad']-s['view_azimuth_rad']),math.cos(r['view_azimuth_rad']-s['view_azimuth_rad']))) for s in selected)
   selected.append(max(avail,key=lambda r:(.05+distance(r))*math.sqrt(r['bbox_area'])))
  if len(selected)<6:continue
  spread=max(abs(math.atan2(math.sin(r['view_azimuth_rad']-s['view_azimuth_rad']),math.cos(r['view_azimuth_rad']-s['view_azimuth_rad']))) for r in selected for s in selected)
  score=math.sqrt(np.median([r['bbox_area'] for r in selected[:6]]))*min(spread,2.5)
  ranked.append(dict(laser_object_id=lid,scene=scene,context=d['context'],raw_tfrecord=d['raw_tfrecord'],score=score,spread_degrees=math.degrees(spread),views=selected))
 ranked.sort(key=lambda x:x['score'],reverse=True);ranked=ranked[:5]
 byframe=defaultdict(list)
 for rank,t in enumerate(ranked):
  t['candidate_id']=f'{scene}_{rank}'
  for j,r in enumerate(t['views']):byframe[r['frame_index']].append((t,j,r))
 for fi,frame in frames(Path(d['raw_tfrecord'])):
  if fi not in byframe:continue
  ims={}
  for t,j,r in byframe[fi]:
   c=r['camera']
   if c not in ims:ims[c]=Image.open(io.BytesIO(next(x.image for x in frame.images if x.name==c))).convert('RGB')
   im=ims[c];b=r['box_xyxy'];pad=max(b[2]-b[0],b[3]-b[1])*.15
   crop=[max(0,int(b[0]-pad)),max(0,int(b[1]-pad)),min(im.width,int(b[2]+pad)),min(im.height,int(b[3]+pad))]
   dest=out/f'{t["candidate_id"]}_{j:02}.png';im.crop(crop).save(dest)
   r.update(path=str(dest),crop_box=crop,source_type='real',raw_tfrecord=d['raw_tfrecord'],context=d['context'],laser_object_id=t['laser_object_id'],box_crop_xyxy=[b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]],setup='metadata-assisted',association_method='native projected box IoU >0.4; margin >0.15')
 sheet=Image.new('RGB',(1400,250*len(ranked)),'#ddd');draw=ImageDraw.Draw(sheet)
 for i,t in enumerate(ranked):
  for j,r in enumerate(t['views'][:5]):
   x=j*280;y=i*250;im=ImageOps.contain(Image.open(r['path']),(280,205));sheet.paste(im,(x,y+35));draw.text((x+3,y+3),f'{i}/{j} f{r["frame_index"]} c{r["camera"]} az{math.degrees(r["view_azimuth_rad"]):.0f}',fill='black')
 sheet.save(out/f'{scene}_candidates.jpg');all_tracks.extend(ranked)
 print(scene,[(t['candidate_id'],round(t['score']),round(t['spread_degrees'])) for t in ranked],flush=True)
(out/'candidates.json').write_text(json.dumps(all_tracks,indent=2))
