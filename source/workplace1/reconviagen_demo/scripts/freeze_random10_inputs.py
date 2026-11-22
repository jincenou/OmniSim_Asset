"""Freeze input-only reviewed random proposals before any method output."""
from pathlib import Path
import json, hashlib, shutil, math
import numpy as np
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
specs=[
 ('selection',1,'bluegrey_suv','suv',[2,5,3,1],'front','Rear training view is partly occluded; source front is clear.'),
 ('selection',2,'white_pickup','pickup',[0,1,3,4],'rear','Four clear training views; heldout views contain poles.'),
 ('selection',3,'silver_minivan','minivan',[1,0,4,5],'front','Four clear training views; two heldout observations are occluded.'),
 ('selection',7,'white_suv','suv',[0,4,5,3],'rear','Some rear views are soft; heldout observations are occluded.'),
 ('selection',12,'silver_crossover','suv',[1,0,3,5],'front','One training view has foreground vehicle occlusion at the bottom.'),
 ('selection',14,'blue_sedan','sedan',[1,0,2,4],'front','Adjacent white SUV partially occludes the front in some views.'),
 ('selection',15,'grey_minivan','minivan',[0,1,2,4],'front','Clear vehicle; sunlight and windshield reflections vary.'),
 ('extension',0,'white_sedan','sedan',[0,1,3,5],'rear','Limited rear-quarter coverage; distant views are softer.'),
 ('extension',2,'dark_sedan','sedan',[0,1,3,4],'rear','Limited rear-quarter coverage; distant views are softer.'),
 ('extension',3,'grey_electric_suv','suv',[1,0,2,4],'front','Limited front-quarter coverage; heldout views partly occluded.'),
]
registries={k:json.loads((ROOT/f'inputs/random10_{k}/candidates.json').read_text()) for k in ['selection','extension']}
frozen=ROOT/'inputs/random10_cases.json'
if frozen.exists():raise FileExistsError(frozen)
cases=[];handoff=[]
for i,(group,rank,name,subtype,train,end,note) in enumerate(specs):
 c=registries[group]['candidates'][rank];case_id=f'random10_{i+1:02}_{name}';out=ROOT/'inputs'/case_id
 if out.exists():raise FileExistsError(out)
 out.mkdir();images=[]
 for j in train+[j for j in range(6) if j not in train]:
  r=dict(c['views'][j]);rid=f'real_f{r["frame_index"]:03}_c{r["camera"]}';dest=out/(rid+'.png');shutil.copyfile(r['path'],dest)
  r.update(id=rid,path=str(dest),split='train' if j in train else 'heldout',source_type='real',context=c['context'],raw_tfrecord=c['raw_tfrecord'],laser_object_id=c['laser_object_id'],association_method='native projected 3D laser box IoU>=0.4 and margin>=0.15; input-only visual identity review',proposal_view=j)
  assert hashlib.sha256(dest.read_bytes()).hexdigest()==r['sha256'];images.append(r)
 dims=np.median([r['projected_laser_box'][3:6] for r in images if r['split']=='train'],axis=0).tolist()
 case=dict(case_id=case_id,original_obj_id=c['laser_object_id'],role=note,subtype=subtype,setup='metadata-assisted',images=images,metric_dimensions_lwh=dims,scale_source='median native laser boxes of training views only',identity=dict(context=c['context'],laser_object_id=c['laser_object_id']),flux_reference_id=images[0]['id'],visible_end=end,selection_method='fixed random seed 20261023, input-only review, 7 initial proposals plus first 3 passing supplementary stream',proposal_group=group,proposal_rank=rank,azimuth_span_degrees=c['spread_degrees'],heldout_policy='Excluded from generation, FLUX, refinement, registration, dimensions and parameter selection; occluded heldouts are not unqualified geometry ground truth',scope_note='10-identity local Waymo pilot from three scenes; not a replication of the 128-scene paper benchmark; native 3D metadata assists identity/scale/camera')
 (out/'manifest.json').write_text(json.dumps(case,indent=2));cases.append(case)
 sheet=Image.new('RGB',(1200,600),'#ddd');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(images):
  x=j%3*400;y=j//3*300;im=ImageOps.contain(Image.open(r['path']),(400,260));sheet.paste(im,(x,y+35));draw.text((x+3,y+4),r['id']+' '+r['split']+(' SOURCE' if j==0 else ''),fill='black')
 sheet.save(out/'real_inputs.jpg',quality=92)
 handoff.append(dict(scene_id=c['context'],obj_id=case_id,category='car',image_path=str(out/'flux_reference.png'),prompts=[f'Generate a clear {end}-view image of this car.','Generate a clear side-view image of this car.']))
frozen.write_text(json.dumps(cases,indent=2))
(ROOT/'inputs/random10_flux_manifest.json').write_text(json.dumps(handoff,indent=2))
(ROOT/'inputs/random10_experiment_matrix.json').write_text(json.dumps([dict(case_id=c['case_id'],variant=v,seed=42,status='inputs_frozen') for c in cases for v in ['B0','R','F','H','B0_real1','B0_real4']],indent=2))
(ROOT/'inputs/random10_query_registry.json').write_text(json.dumps(dict(seed=20261023,query_count=10,query_ids=[c['case_id'] for c in cases],frozen_cases_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),selection='Before any new FLUX or 3D generation. Original stream reviewed in input_review.json. Extension ranks 0,2,3 accepted; rank1 rejected because of persistent foreground-vehicle occlusion. Later extension ranks unused.',aor_denominator='All ten frozen queries for each route, including subsequent generation/export/render failures.',source_conditioned_subset='Intersection of valid outputs across all six main routes; source view fixed to flux_reference_id.',prompt_policy='Original simple vehicle end/side templates passed verbatim. End chosen from observed source. No Qwen attribute call or additional semantic rewrite in this pilot.'),indent=2))
print('Frozen',len(cases),'new identities and',len(cases)*6,'primary runs')
