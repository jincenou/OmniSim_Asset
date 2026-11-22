import asyncio,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import runtime
from asset_extraction.config import load_config
from asset_extraction.io_utils import read_json,write_json
from asset_extraction.quality import reassess
os.environ['VLM_BASE_URL']='http://127.0.0.1:8082/v1'
os.environ['VLM_RESPONSE_FORMAT']='json_object'
WORK=Path(__file__).resolve().parent
IDS={'scene-0063_t000126','scene-0063_t001240','waymo_val_030_t000433',
     'waymo_val_030_t000759','scene-0063_t002286','scene-0063_t001932',
     'scene-0063_t001956','scene-0063_t001942'}

async def one(chunk):
    state=read_json(chunk/'curation/curation.json')
    subset={k:v for k,v in state['assets'].items() if k in IDS}
    if not subset:return []
    state=dict(state,assets=subset)
    generation=read_json(chunk/'generation/generation.json')
    ds='nuscenes' if chunk.name.startswith('nuscenes') else 'waymo'
    scene=next(iter(subset.values()))['entry']['provenance']['scene_id']
    config=load_config(ROOT/'configs'/f'{ds}_{scene}.yaml')
    reviewed=await reassess(state,generation,config,WORK/(chunk.name+'_validated.json'))
    return [dict(uid=uid,chunk=str(chunk),before=state['assets'][uid]['status'],after=x['status'],
        audit=x['entry']['quality']['audit']) for uid,x in reviewed['assets'].items()]

async def main():
    results=await asyncio.gather(*(one(f.parent) for f in sorted((ROOT/'chunks').glob('*/completed.json'))))
    rows=[r for rr in results for r in rr]
    write_json(WORK/'pilot_results_validated.json',dict(assets=rows,production_changed=False))
    for r in rows:print(json.dumps(r,ensure_ascii=False),flush=True)

asyncio.run(main())
