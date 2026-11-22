"""Replay only confirmed libx264 failures; retain originals and reuse generation."""
import asyncio,copy,fcntl,json,os,sys,time,traceback
from pathlib import Path
from collections import Counter
ROOT=Path('/mnt/ojc/workplace1/asset_work/scene_production_20261031')
WORK=Path(__file__).parent
sys.path.insert(0,str(ROOT))
os.environ.update(VLM_BASE_URL='http://127.0.0.1:8082/v1',VLM_RESPONSE_FORMAT='json_object',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from runtime import SharedGpuPool,apply_identity_curation,read_json,write_json,sha256,LIBRARY
from asset_extraction import generation
from asset_extraction.export import curate
from asset_extraction.quality import repair_shapes
from asset_extraction.bank import publish
from asset_extraction.config import load_config
SLOTS=asyncio.Semaphore(2)
class LimitedPool(SharedGpuPool):
 async def acquire(self,stage):
  await SLOTS.acquire()
  try:return await super().acquire(stage)
  except BaseException:SLOTS.release();raise
 def release(self,gpu):
  try:super().release(gpu)
  finally:SLOTS.release()
generation.GpuPool=LimitedPool
GPUS=[0,1,2,3,5,6]
def candidate(chunk):
 c=read_json(chunk/'curation/curation.json');ids=[]
 for uid,a in c['assets'].items():
  error=a.get('error','')
  if a['status']!='failed' or not error.startswith('RuntimeError: Render failed: '):continue
  log=Path(error.split('RuntimeError: Render failed: ',1)[1])
  if log.is_file() and "Unknown encoder 'libx264'" in log.read_text(errors='replace'):ids.append(uid)
 return ids
async def one(chunk,ids):
 work=chunk/'encoder_recovery_20261103';work.mkdir(exist_ok=True)
 if (work/'completed.json').exists():return read_json(work/'completed.json')
 for src,name in [('completed.json','previous_completed.json'),('curation/curation.json','previous_curation.json'),('technical_validity.json','previous_technical_validity.json'),('selected_generation.json','previous_selected_generation.json')]:
  if not (work/name).exists():write_json(work/name,read_json(chunk/src))
 old=read_json(work/'previous_curation.json');receipt=read_json(work/'previous_completed.json');selected=read_json(work/'previous_selected_generation.json')
 assert all(old['assets'][uid]['status']=='failed' for uid in ids)
 subset=copy.deepcopy(selected);subset['assets']=[a for a in selected['assets'] if a['obj_id'] in ids]
 assert len(subset['assets'])==len(ids)
 subset['counts']=dict(queries=len(ids),generated=sum(a['status']=='generated' for a in subset['assets']))
 subset['recovery']=dict(reason='libx264 unavailable in PATH ffmpeg',source_generation=str(chunk/'selected_generation.json'),original_query_ids=ids)
 path=work/'generation.json'
 if not path.exists():write_json(path,subset)
 else:assert read_json(path)==subset
 config=load_config(ROOT/'configs'/f"{receipt['dataset']}_{subset['assets'][0]['scene_id']}.yaml")
 statepath=work/'curation/curation.json'
 state=read_json(statepath) if statepath.exists() else {}
 if 'counts' not in state:state=await curate(path,work/'library',config,renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=GPUS,work_dir=work/'curation')
 final,regen=await repair_shapes(path,state,config,work/'shape_retry',renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=GPUS)
 merged=copy.deepcopy(old);merged['assets'].update(final['assets'])
 existing={a['source_obj_id'] for a in receipt['published']}
 merged=apply_identity_curation(merged,receipt['dataset'],ROOT.name,protected_ids=existing)
 for uid in existing:merged['assets'][uid]=copy.deepcopy(old['assets'][uid])
 for uid in ids:
  a=merged['assets'][uid];a['encoder_recovery']=dict(previous_record=str(work/'previous_curation.json'),original_query_identity_unchanged=True,render_code_sha256=sha256('/mnt/ojc/DriveLab/asset_extraction/export.py'))
  if 'entry' in a:a['entry']['provenance']['batch_id']=ROOT.name
 merged['counts']=dict(queries=len(merged['assets']),**Counter(a['status'] for a in merged['assets'].values()))
 added=publish(dict(assets={uid:merged['assets'][uid] for uid in ids}),LIBRARY,dataset=receipt['dataset'],scene_id=subset['assets'][0]['scene_id'])
 receipt['published'] += [dict(uid=a['uid'],canonical_path=str(LIBRARY/a['canonical']['canonical_path']),sha256=a['canonical']['sha256'],source_obj_id=a['provenance']['source_obj_id'],status=a['status'],category=a['category']) for a in added]
 replacements={a['obj_id']:a for a in regen['assets']}
 selected['assets']=[replacements.get(a['obj_id'],a) for a in selected['assets']]
 tech=read_json(work/'previous_technical_validity.json')
 for t in tech['assets']:
  if t['obj_id'] not in ids:continue
  a=merged['assets'][t['obj_id']];render=a.get('entry',{}).get('quality',{}).get('rendering',{})
  row=replacements[t['obj_id']]
  if not render:
   for kind in ['canonical_render','axis_render']:
    q=work/'curation'/f"asset_{row['item_index']:05d}"/kind/'render.json'
    if q.exists():render=read_json(q);break
  pix=render.get('foreground_pixels',[])
  t.update(status=a['status'],rendering=render,valid_16=bool(row['status']=='generated' and row.get('gaussian_count',0)>0 and render.get('all_16_nonempty') and len(pix)==16 and all(v>0 for v in pix)))
 receipt.update(counts=merged['counts'],kept=sum(a['status']=='keep' for a in receipt['published']),encoder_recovery=dict(previous_receipt=str(work/'previous_completed.json'),updated=time.time(),queries=len(ids)))
 assert receipt['queries']==read_json(work/'previous_completed.json')['queries']
 # Commit completed receipt last so maintenance rebuilds the ledger from final state.
 write_json(chunk/'selected_generation.json',selected);write_json(chunk/'curation/curation.json',merged);write_json(chunk/'technical_validity.json',tech);write_json(chunk/'completed.json',receipt)
 result=dict(chunk=str(chunk),queries=len(ids),counts=dict(Counter(merged['assets'][uid]['status'] for uid in ids)),published=len(added),completed=time.time())
 write_json(work/'completed.json',result);return result
async def main():
 with (WORK/'recovery.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=[]
  for p in sorted((ROOT/'chunks').glob('*/completed.json')):
   ids=candidate(p.parent)
   if ids:plan.append(dict(chunk=str(p.parent),ids=ids))
  write_json(WORK/'plan.json',dict(created=time.time(),queries=sum(len(a['ids']) for a in plan),chunks=plan))
  results=[]
  for a in plan:
   write_json(WORK/'status.json',dict(status='running',pid=os.getpid(),current_chunk=a['chunk'],completed_chunks=len(results),updated=time.time()))
   try:result=await one(Path(a['chunk']),a['ids'])
   except Exception as exc:
    traceback.print_exc();result=dict(chunk=a['chunk'],error=f'{type(exc).__name__}: {exc}')
   results.append(result);write_json(WORK/'results.json',results);print(json.dumps(result),flush=True)
  write_json(WORK/'status.json',dict(status='completed_with_errors' if any('error' in a for a in results) else 'completed',pid=os.getpid(),results=results,updated=time.time()))
if __name__=='__main__':asyncio.run(main())
