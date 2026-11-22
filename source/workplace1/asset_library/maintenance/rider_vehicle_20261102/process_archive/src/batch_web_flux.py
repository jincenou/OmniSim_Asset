"""Batch reviewed singleton jobs to amortize model load; scatter evidence back to identities."""
import asyncio,copy,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.generation import digest,GpuPool,_launch,checked_outputs
from asset_extraction.io_utils import write_json
async def main():
 queue=json.loads((W/'sources/web_expansion_20261103/queue.json').read_text())
 names=[r['name'] for r in queue if r['status']=='ready_for_flux']
 if not names:raise ValueError('No approved inputs')
 work=W/'runs/web_expansion_20261103_flux';work.mkdir(exist_ok=True)
 originals=[json.loads((W/'runs'/n/'F/jobs.json').read_text()) for n in names]
 rows=[]
 for i,b in enumerate(originals):
  r=copy.deepcopy(b['assets'][0]);r.update(item_index=i,flux_seed_index=0);rows.append(r)
 bundle=dict(config=originals[0]['config'],assets=rows);bundle['signature']=digest(bundle)
 jobs=work/'jobs.json';statepath=work/'generation.json'
 if statepath.exists():
  state=json.loads(statepath.read_text());assert state['signature']==bundle['signature']
 else:state=dict(schema_version=1,signature=bundle['signature'],config=bundle['config'],assets=[dict(r,stages={},status='pending') for r in rows])
 write_json(jobs,bundle);write_json(statepath,state)
 def accept(result):
  i=result['item_index'];r=state['assets'][i]
  if result['status']=='succeeded':checked_outputs(result)
  r['stages']['flux']=result;r['status']=result['status'];write_json(statepath,state)
  local=copy.deepcopy(result);local['item_index']=0
  single=dict(schema_version=1,signature=originals[i]['signature'],config=originals[i]['config'],assets=[dict(originals[i]['assets'][0],stages=dict(flux=local),status=local['status'])])
  write_json(W/'runs'/names[i]/'F/generation.json',single)
 pool=GpuPool([1,3,5,6])
 ids=[i for i,r in enumerate(state['assets']) if r['stages'].get('flux',{}).get('status')!='succeeded']
 async def chunk(indices):
  if not indices:return
  gpu=await pool.acquire('flux')
  try:await _launch('flux',indices,gpu,jobs,accept)
  finally:pool.release(gpu)
 await asyncio.gather(*(chunk(ids[i::2]) for i in range(2)))
 state['status']='awaiting_flux_visual_review';write_json(statepath,state)
 print('BATCH_FLUX_DONE',sum(r['status']=='succeeded' for r in state['assets']),flush=True)
asyncio.run(main())
