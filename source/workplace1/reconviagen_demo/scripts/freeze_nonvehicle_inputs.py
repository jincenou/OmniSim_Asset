from pathlib import Path
import json,hashlib,shutil
from PIL import Image,ImageDraw,ImageOps
R=Path(__file__).resolve().parents[1];O=R/'nonvehicle10';candidates=json.loads((O/'inputs/nusc_candidates.json').read_text())
for ci,name in [(0,'nonvehicle_09_concrete_barrier'),(1,'nonvehicle_10_orange_cone')]:
 c=candidates[ci];obs=sorted(c['observations'],key=lambda r:r['area'],reverse=True)[:4];out=O/'inputs'/name;out.mkdir(exist_ok=True);rows=[]
 for i,r in enumerate(obs):
  im=Image.open(r['image_path']).convert('RGB');b=r['box_xyxy'];pad=.12*max(b[2]-b[0],b[3]-b[1]);crop=[max(0,int(b[0]-pad)),max(0,int(b[1]-pad)),min(im.width,int(b[2]+pad)),min(im.height,int(b[3]+pad))];fp=out/f'real_{i:02}.png';im.crop(crop).save(fp);rows.append(dict(r,id=f'real_{i:02}',source_type='real',split='train',path=str(fp),sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),crop_box=crop,box_crop_xyxy=[b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]],association={'method':'same native nuScenes instance token and camera; projected annotated cuboid'}))
 Image.open(obs[0]['image_path']).convert('RGB').save(out/'source_full.jpg',quality=92)
 case=dict(case_id=name,category=c['category'],dataset='nuscenes',original_obj_id=c['instance_token'],flux_reference_id='real_00',images=rows,selection='Native annotated instance, verified before generation, four distinct observations',matched_source_pose_available=False,scale_source='normalized-only comparison; native cuboid recorded but not used for synthesis',setup='Native category and 3D-box projection + SAM2; same-camera native instance association')
 (out/'manifest.json').write_text(json.dumps(case,indent=2))
paths=sorted((O/'inputs').glob('nonvehicle_*/manifest.json'));cases=[json.loads(p.read_text()) for p in paths];assert len(cases)==10
original=json.loads(Path('/mnt/wmj/workspace4/pipeline/json/rigid_5k_manifest.json').read_text());templates={r['category']:r['prompts'] for r in original};templates.update(trash_can=['Complete the missing parts of this trash can and make it clear. Preserve its color and shape.'],fire_hydrant=['Complete the missing parts of this fire hydrant and make it clear. Preserve its color and shape.'])
items=[];sheet=Image.new('RGB',(1400,250*len(cases)),'#ddd');d=ImageDraw.Draw(sheet)
for i,c in enumerate(cases):
 items.append(dict(obj_id=c['case_id'],image_path=str(O/'inputs'/c['case_id']/'flux_reference.png'),category=c['category'],prompts=templates[c['category']]))
 d.text((3,i*250+3),c['case_id']+f"  ({len(c['images'])} real)",fill='black')
 for j,r in enumerate(c['images']):
  im=ImageOps.contain(Image.open(r['path']),(330,220));sheet.paste(im,(j*350,i*250+25))
(O/'inputs/cases.json').write_text(json.dumps(cases,indent=2));(O/'inputs/fresh_flux_manifest.json').write_text(json.dumps(items,indent=2));sheet.save(O/'docs/frozen_sources.jpg',quality=92)
(O/'docs/input_freeze.json').write_text(json.dumps(dict(cases=[dict(case_id=c['case_id'],category=c['category'],real_count=len(c['images']),source_sha256=c['images'][0]['sha256']) for c in cases],prompts='Exact original category prompts where available; explicit equivalent prompts for added trash_can and fire_hydrant categories',manifest_sha256=hashlib.sha256((O/'inputs/fresh_flux_manifest.json').read_bytes()).hexdigest()),indent=2));print('FROZEN',len(cases),flush=True)
