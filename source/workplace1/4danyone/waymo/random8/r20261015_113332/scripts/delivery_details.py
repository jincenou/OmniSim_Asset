"""CPU-only, shared body-region previews and comparable asset-size accounting."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,csv,asyncio,shutil,uuid
import numpy as np
from PIL import Image,ImageDraw,ImageOps
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parents[2]/'repo'))
from fdanyone.skeleton.keypoints import KEYPOINT_NAMES
from comparison_output import complete_results
B=['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61'];REGIONS={'left_hand':[i for i,n in enumerate(KEYPOINT_NAMES) if n=='left-wrist' or (n.startswith('left-') and n.endswith('-tip') and 'toe' not in n)],'right_hand':[i for i,n in enumerate(KEYPOINT_NAMES) if n=='right-wrist' or (n.startswith('right-') and n.endswith('-tip') and 'toe' not in n)],'feet':[i for i,n in enumerate(KEYPOINT_NAMES) if any(x in n for x in ['ankle','toe','heel'])]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 t=p.with_suffix(p.suffix+'.writing');t.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n');t.replace(p)
def sizes(label,branch):
 base=R/'samples'/label;root=base/'evaluations'/branch;asset=json.loads((root/'asset_manifest.json').read_text());ft=branch.startswith('ftgs');model=base/branch if ft else R/'four_dg/runs'/f'{label}_grid{branch[-2:]}';done=json.loads((model/('training_complete.json' if ft else 'complete.json')).read_text());paths=[Path(asset['dynamic_ply'])] if ft else [Path(p) for p in asset['dynamic_files']];checkpoints=sorted((model/'ckpts').glob('*.pt')) if ft else sorted(model.glob('chkpnt_*.pth.gz'));payload={str(p):p.stat().st_size for p in paths};resume={str(p):p.stat().st_size for p in checkpoints}
 row={'label':label,'branch':branch,'dynamic_payload_bytes':sum(payload.values()),'all_retained_recovery_checkpoint_bytes':sum(resume.values()),'initial_gaussians':done['initial_gaussians'],'final_gaussians':done['final_gaussians'],'deformation_network_parameters':0 if ft else done['network_parameters'],'full_training_seconds':done['elapsed_seconds'],'peak_allocated_gib':done['peak_allocated_gib'],'total_run_artifact_bytes':sum(p.stat().st_size for p in model.rglob('*') if p.is_file())};write(root/'asset_sizes.json',{'row':row,'dynamic_payload_files':payload,'recovery_checkpoint_files':resume,'definition':'Payload is serialized render representation only:FTGSdynamicPLY orHUSTcanonicalPLY+deformation/table/accum. Resume optimizer checkpoints counted separately. Runtime/code/sharedinput excluded from payload;total_run_artifact_bytes includes all saved snapshots/logs.','metric_scope':'Full backend recipes differ in representation/capacity;file size is not matched-capacity quality comparison.'});return row

def parts(label,branches):
 if branches!=B:return
 base=R/'samples'/label
 results,missing=complete_results(base)
 if missing:return
 directory=base/'comparison';signature=hashlib.sha256(''.join(sha(base/'evaluations'/b/'results.json') for b in B).encode()).hexdigest()[:16];final=directory/'details';manifest=final/'manifest.json'
 if manifest.exists():
  existing=json.loads(manifest.read_text())
  if existing.get('input_results_signature')==signature and all(Path(row['file']).exists() and sha(Path(row['file']))==row['sha256'] for row in existing.get('images',[])) and len(existing.get('images',[]))==9:return
 staging=directory/('.parts-build-'+uuid.uuid4().hex);staging.mkdir(parents=True)
 try:
  build_parts(label,branches,signature,staging,final)
  final.mkdir(exist_ok=True)
  for path in staging.glob('*.jpg'):path.replace(final/path.name)
  (staging/'manifest.json').replace(manifest)
 finally:shutil.rmtree(staging)

def build_parts(label,branches,signature,out,final):
 base=R/'samples'/label;source=base/'generation/preflight/conditioning';points=np.load(source/'keypoints_3d.npy');assert points.shape[1:]==(70,3);frames=len(points);cameras=json.loads((source/'cameras.json').read_text())['cameras'];manifest=out/'manifest.json';spec=[];files=[]
 for view in [0,8,16]:
  camera=cameras[view];w=np.linalg.inv(camera['camera_to_world']);K=np.asarray(camera['K']).copy();K[:2]*=.5;projected=points@w[:3,:3].T+w[:3,3];q=projected@K.T;uv=q[...,:2]/q[...,2:];assert np.isfinite(uv).all()
  for f in [0,(frames-1)//2,frames-1]:
   name=f'view{view:02d}_frame{f:03d}.jpg';pairs={b:Image.open(base/'evaluations'/b/name).convert('RGB') for b in branches};target=pairs[branches[0]].crop((0,0,352,640));mask=np.asarray(Image.open(base/'data/images'/f'cam{view:02d}'/f'{f:06d}.png'))[...,3]>127;ys,_=np.where(mask);pad=max(12,int(round(.04*(ys.max()-ys.min()+1))));canvas=Image.new('RGB',(5*256,3*280+64),(25,25,25));draw=ImageDraw.Draw(canvas);titles=['Generated reference',*B]
   for col,title in enumerate(titles):draw.text((col*256+5,5),title,fill='white')
   for row,(region,ids) in enumerate(REGIONS.items()):
    v=uv[f,ids];lo=np.floor(v.min(0)-pad).astype(int);hi=np.ceil(v.max(0)+pad).astype(int);lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[352,640]);assert (hi>lo).all();box=[*lo.tolist(),*hi.tolist()];spec.append({'view':view,'frame':f,'region':region,'xyxy':box,'keypoint_names':[KEYPOINT_NAMES[i] for i in ids],'padding_pixels':pad})
    panes=[target]+[pairs[b].crop((352,0,704,640)) for b in B]
    for col,pane in enumerate(panes):
     y=24+row*280;draw.text((col*256+5,y),region+' pose proxy',fill='white')
     patch=pane.crop(box);scale=min(246/patch.width,246/patch.height);patch=patch.resize((max(1,round(patch.width*scale)),max(1,round(patch.height*scale))),Image.Resampling.NEAREST);canvas.paste(patch,(col*256+(256-patch.width)//2,y+24+(246-patch.height)//2))
   draw.text((5,canvas.height-18),'Conditioning-pose proxies;visual only. Accessory location unknown;not scored. Display enlargement adds no detail.',fill='white');path=out/f'parts_view{view:02d}_frame{f:03d}.jpg';canvas.save(path,quality=95);files.append({'file':str(final/path.name),'sha256':sha(path)})
 write(manifest,{'input_results_signature':signature,'state':'uniform_pose_proxy_regions_visualized','label':label,'branches':branches,'region_provider':'same existing root-local MHR70conditioning projected by actual generated cameraK/c2w;independent of each backendprediction','not_semantic_GT':True,'no_new_part_model':True,'no_part_metrics_from_JPEG':True,'accessory_region':'unknown_not_localized_not_scored','source_precision':'Existing visualizationJPEG pairs;not inputs to quantitative metrics. Same rectangles across all methods;no target-mask clipping of predictedforeground.','specification':spec,'images':files,'conditioning_points_sha256':sha(source/'keypoints_3d.npy')})

def main():
 rows=[];groups={};errors=[]
 for p in (R/'samples').glob('*/evaluations/*/results.json'):
  if 'snapshot' in p.parent.name or p.parent.name not in B:continue
  label=p.parents[2].name;branch=p.parent.name
  try:rows.append(sizes(label,branch));groups.setdefault(label,[]).append(branch)
  except Exception as e:errors.append({'file':str(p),'error':str(e)})
 for label,branches in groups.items():
  try:parts(label,[b for b in B if b in branches])
  except Exception as e:errors.append({'label':label,'parts_error':str(e)})
 write(R/'resource_table.json',{'rows':rows,'errors':errors,'updated_at':datetime.now(timezone.utc).isoformat(),'scope':'Comparable serializeddynamicpayload and recoverycheckpoints separately;actual fullrecipe resource statistics'})
 if rows:
  with (R/'resource_table.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
async def watch():
 while True:
  main()
  if json.loads((R/'STATUS.json').read_text())['state'].startswith('all_registered_tasks_terminal'):return
  await asyncio.sleep(120)
if __name__=='__main__':asyncio.run(watch()) if '--watch' in sys.argv else main()
