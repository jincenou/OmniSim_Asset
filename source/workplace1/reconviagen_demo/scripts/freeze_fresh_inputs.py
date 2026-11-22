"""Freeze three visually selected identities before FLUX or reconstruction."""
from native_candidates import ROOT,frames
from collections import defaultdict
from pathlib import Path
import json,math,io,hashlib
import numpy as np
from PIL import Image,ImageOps,ImageDraw

selections=[
 dict(case_id='fresh_blue_suv',lid='U3BxS83wGKcsx-iHkptuaw',subtype='suv',train=[(110,2),(118,2),(124,1),(134,3)],heldout=[(114,2),(129,1)],reference=(118,2),role='clear moving dark blue SUV with roof crossbars; front-side-rear coverage'),
 dict(case_id='fresh_black_pickup',lid='K-P8imHLHIcMZ5c4gAB2ZQ',subtype='pickup',train=[(143,2),(152,2),(158,1),(169,3)],heldout=[(148,2),(163,1)],reference=(143,2),role='moving black extended-cab pickup with open bed and raised suspension'),
 dict(case_id='fresh_silver_suv',lid='fR1DGWe-Sn4gEsBrsbsS4w',subtype='suv',train=[(136,2),(141,2),(148,1),(159,3)],heldout=[(145,1),(153,1)],reference=(136,2),role='moving silver boxy SUV; rear training view has a pole occlusion'),
]
d=json.loads((ROOT/'inputs/silver_sedan_03/associated_inventory.json').read_text());byframe=defaultdict(list)
for s in selections:
 out=ROOT/'inputs'/s['case_id'];out.mkdir(exist_ok=True);s['images']=[]
 for split in ['train','heldout']:
  for fi,cam in s[split]:
   rows=[r for r in d['observations'] if r.get('projected_laser_object_id')==s['lid'] and r['frame_index']==fi and r['camera']==cam];assert len(rows)==1
   r=dict(rows[0]);box=r['projected_laser_box'];c=np.array(r['camera_to_ego']).reshape(4,4)[:3,3];delta=c[:2]-np.array(box[:2]);a=math.atan2(delta[1],delta[0])-box[6]
   r.update(id=f'real_f{fi:03}_c{cam}',split=split,source_type='real',context=d['context'],raw_tfrecord=d['raw_tfrecord'],laser_object_id=s['lid'],view_azimuth_rad=math.atan2(math.sin(a),math.cos(a)),association_method='native projected laser box IoU >0.4 and margin >0.15; visual identity review')
   s['images'].append(r);byframe[fi].append((s,r))
for fi,frame in frames(Path(d['raw_tfrecord'])):
 if fi not in byframe:continue
 for s,r in byframe[fi]:
  im=Image.open(io.BytesIO(next(x.image for x in frame.images if x.name==r['camera']))).convert('RGB');b=r['box_xyxy'];pad=max(b[2]-b[0],b[3]-b[1])*.15
  crop=[max(0,int(b[0]-pad)),max(0,int(b[1]-pad)),min(im.width,int(b[2]+pad)),min(im.height,int(b[3]+pad))]
  dest=ROOT/'inputs'/s['case_id']/(r['id']+'.png');im.crop(crop).save(dest)
  r.update(path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),crop_box=crop,box_crop_xyxy=[b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]])
cases=[]
for s in selections:
 out=ROOT/'inputs'/s['case_id'];dims=np.median([r['projected_laser_box'][3:6] for r in s['images'] if r['split']=='train'],axis=0).tolist()
 case=dict(case_id=s['case_id'],original_obj_id=s['lid'],role=s['role'],subtype=s['subtype'],setup='metadata-assisted',images=s['images'],metric_dimensions_lwh=dims,scale_source='median native laser boxes of training views only',identity=dict(context=d['context'],laser_object_id=s['lid']),flux_reference_id=f'real_f{s["reference"][0]:03}_c{s["reference"][1]}',selection_method='native track association and input-only visual review',heldout_policy='Excluded from FLUX, generation, refinement, registration, size fitting and parameter choice',scope_note='Fresh selection authorized by user; three clearer moving vehicles replace two weak old cases; old service van retained as diagnostic')
 (out/'manifest.json').write_text(json.dumps(case,indent=2));cases.append(case)
 sheet=Image.new('RGB',(1200,600),'#ddd');dr=ImageDraw.Draw(sheet)
 for j,r in enumerate(case['images']):
  x=j%3*400;y=j//3*300;im=ImageOps.contain(Image.open(r['path']),(400,255));sheet.paste(im,(x,y+35));dr.text((x+3,y+3),f'{r["id"]} {r["split"]}',fill='black')
 sheet.save(out/'real_inputs.jpg')
 print(case['case_id'],dims,flush=True)
(ROOT/'inputs/fresh_cases.json').write_text(json.dumps(cases,indent=2))
(ROOT/'inputs/fresh_experiment_matrix.json').write_text(json.dumps([dict(case_id=c['case_id'],variant=v,seed=42,status='inputs_frozen') for c in cases for v in ['B0','R','F','H']],indent=2))
