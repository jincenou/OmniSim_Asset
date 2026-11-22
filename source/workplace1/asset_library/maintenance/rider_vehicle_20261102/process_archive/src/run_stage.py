"""Run one bounded, leased stage; allow inspection between generation and reconstruction."""
import argparse,asyncio,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'DriveLab';sys.path.insert(0,str(ROOT))
from asset_extraction.generation import GpuPool,_launch,checked_outputs
from asset_extraction.io_utils import read_json,write_json
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('stage',choices=['flux','H','B0']);p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
async def main():
 jobs=a.work/'jobs.json';bundle=read_json(jobs);statepath=a.work/'generation.json'
 state=read_json(statepath) if statepath.exists() else dict(schema_version=1,signature=bundle['signature'],config=bundle['config'],assets=[dict(row,stages={},status='pending') for row in bundle['assets']])
 if state['signature']!=bundle['signature']:raise ValueError('State and jobs signature differ')
 def accept(result):
  row=state['assets'][result['item_index']]
  if result['status']=='succeeded':checked_outputs(result)
  row['stages'][a.stage]=result;row['status']='generated' if a.stage!='flux' and result['status']=='succeeded' else result['status']
  if row['status']=='generated':row.update({k:result[k] for k in ['raw_path','raw_sha256','raw_up_axis','gaussian_count']})
  write_json(statepath,state)
 write_json(statepath,state)
 ids=[r['item_index'] for r in state['assets'] if r['stages'].get(a.stage,{}).get('status')!='succeeded']
 if ids:
  pool=GpuPool([a.gpu]);gpu=await pool.acquire(a.stage)
  try:await _launch(a.stage,ids,gpu,jobs,accept)
  finally:pool.release(gpu)
 if any(r['stages'][a.stage]['status']!='succeeded' for r in state['assets']):raise SystemExit(1)
 print('STAGE_DONE',a.stage,flush=True)
asyncio.run(main())
