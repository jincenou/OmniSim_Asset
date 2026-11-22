"""Run reviewed whole-assembly H jobs in shared GPU leases; retain per-identity evidence."""
import asyncio,copy,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.generation import digest,GpuPool,_launch,checked_outputs
from asset_extraction.io_utils import write_json
async def main():
 names=[r['name'] for r in json.loads((W/'sources/web_expansion_20261103/queue.json').read_text()) if r['status']=='ready_for_H']
 if not names:raise ValueError('No reviewed reconstruction inputs')
 work=W/'runs/web_expansion_20261103_H';work.mkdir(exist_ok=True)
 originals=[json.loads((W/'runs'/n/'H0/jobs.json').read_text()) for n in names]
 localstates=[json.loads((W/'runs'/n/'H0/generation.json').read_text()) for n in names]
 rows=[];state_rows=[]
 for i,(b,s) in enumerate(zip(originals,localstates)):
  r=copy.deepcopy(b['assets'][0]);r['item_index']=i;rows.append(r)
  stages=copy.deepcopy(s['assets'][0]['stages']);stages['flux']['item_index']=i
  state_rows.append(dict(r,stages=stages,status='succeeded'))
 bundle=dict(config=originals[0]['config'],assets=rows);bundle['signature']=digest(bundle)
 jobs=work/'jobs.json';statepath=work/'generation.json'
 if statepath.exists():
  state=json.loads(statepath.read_text());assert state['signature']==bundle['signature']
 else:state=dict(schema_version=1,signature=bundle['signature'],config=bundle['config'],assets=state_rows)
 write_json(jobs,bundle);write_json(statepath,state)
 def accept(result):
  i=result['item_index'];r=state['assets'][i]
  if result['status']=='succeeded':checked_outputs(result)
  r['stages']['H']=result;r['status']='generated' if result['status']=='succeeded' else 'failed'
  if r['status']=='generated':r.update({k:result[k] for k in ['raw_path','raw_sha256','raw_up_axis','gaussian_count']})
  write_json(statepath,state)
  local=copy.deepcopy(r);local['item_index']=0
  for stage in local['stages'].values():stage['item_index']=0
  single=dict(schema_version=1,signature=originals[i]['signature'],config=originals[i]['config'],assets=[local]);write_json(W/'runs'/names[i]/'H0/generation.json',single)
 pool=GpuPool([1,3,5,6]);ids=[i for i,r in enumerate(state['assets']) if r['stages'].get('H',{}).get('status')!='succeeded']
 async def chunk(indices):
  if not indices:return
  gpu=await pool.acquire('H')
  try:await _launch('H',indices,gpu,jobs,accept)
  finally:pool.release(gpu)
 await asyncio.gather(*(chunk(ids[i::2]) for i in range(2)))
 state['status']='awaiting_orbit_review';write_json(statepath,state);print('BATCH_H_DONE',sum(r['status']=='generated' for r in state['assets']),flush=True)
asyncio.run(main())
