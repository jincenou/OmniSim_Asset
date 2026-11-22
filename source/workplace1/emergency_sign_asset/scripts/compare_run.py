"""Reuse a verified *actual* FLUX execution for matched-input B0/H comparison."""
import asyncio,copy,json,shutil,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.generation import prepare_jobs,generate,checked_outputs,sha256
from asset_extraction.sign_policy import reviewed_flux_images
from nuscenes.prepare_routes import build_routes
def read(p):return json.loads(p.read_text())
base,name,k,gpu=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]);src=W/'runs'/base;dest=W/'runs'/name;dest.mkdir(exist_ok=True)
original=read(src/'generation/generation.json');old=original['assets'][0]
assert old['stages']['flux']['status']=='succeeded'
checked_outputs(old['stages']['flux']);reviewed_flux_images(src/'generation',old,old['stages']['flux'])
shutil.copyfile(src/'flux_input.json',dest/'flux_input.json')
shutil.copytree(src/'scenes',dest/'scenes',dirs_exist_ok=True)
for p in (dest/'scenes').glob('*/logs/export_items_full.json'):
 records=read(p)
 for row in records:
  row['sign_backend']='H';row['real_images']=row.get('real_images',[])[:k]
  if len(row['real_images'])!=k:raise ValueError('Cannot pad missing real references')
 p.write_text(json.dumps(records,indent=2))
(dest/'routes.json').write_text(json.dumps(build_routes(dest/'flux_input.json',dest),indent=2))
bundle=prepare_jobs(dest/'flux_input.json',dest/'routes.json',read(W/'generation_config.json'));new=bundle['assets'][0]
for key in ('obj_id','scene_id','prompts','image_sha256','sign_metadata_sha256'):
 if old[key]!=new[key]:raise ValueError('Cannot reuse FLUX across changed input: '+key)
for key in ('flux_model','flux_python'):
 if original['config'][key]!=bundle['config'][key]:raise ValueError('Changed FLUX model/runtime')
work=dest/'generation';(work/'asset_00000').mkdir(parents=True,exist_ok=True)
if not (work/'generation.json').exists():
 result=copy.deepcopy(old['stages']['flux'])
 # This is the original succeeded stage receipt with original image hashes/timing,
 # not a real-only substitution or a claim that FLUX executed in this branch.
 result['reuse_provenance']=dict(source_generation=str(src/'generation/generation.json'),source_generation_sha256=sha256(src/'generation/generation.json'),source_signature=original['signature'],execution='reused_verified_completed_stage',comparison='identical F pixels and approved alpha')
 state=dict(schema_version=1,signature=bundle['signature'],config=bundle['config'],assets=[dict(new,stages=dict(flux=result),status='pending')])
 (work/'generation.json').write_text(json.dumps(state,indent=2))
 shutil.copyfile(src/'generation/asset_00000/flux_review.json',work/'asset_00000/flux_review.json')
state=asyncio.run(generate(bundle,work,[gpu]));print(json.dumps(dict(status=state['status'],counts=state['counts'])))
