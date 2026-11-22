"""Real no-real shape retry for the inspected short semi-truck; not a new production query."""
import asyncio,copy,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import runtime
from asset_extraction.quality import repair_shapes,proportion_guard,POLICY
from asset_extraction.config import load_config
from asset_extraction.io_utils import read_json,write_json
os.environ['VLM_BASE_URL']='http://127.0.0.1:8082/v1'
os.environ['VLM_RESPONSE_FORMAT']='json_object'
WORK=Path(__file__).resolve().parent/'retry_smoke'
OID='scene-0063_t002286'

async def main():
    WORK.mkdir(exist_ok=True)
    chunk=ROOT/'chunks/nuscenes_scene-0063_0000'
    generation=read_json(chunk/'generation/generation.json')
    row=next(r for r in generation['assets'] if r['obj_id']==OID)
    generation=dict(generation,assets=[row],counts=dict(queries=1,generated=1))
    write_json(WORK/'initial_generation.json',generation)
    state=read_json(WORK.parent/'nuscenes_scene-0063_0000_validated.json')
    record=copy.deepcopy(state['assets'][OID]);entry=record['entry']
    audit,guard=proportion_guard(row,entry['canonical'],entry['quality']['audit'])
    assert guard['triggered']
    entry['quality'].update(model_audit=entry['quality']['audit'],audit=audit,geometry_guard=guard,admission_policy=POLICY)
    record['status']=entry['status']='reject'
    state=dict(state,assets={OID:record},counts=dict(queries=1,reject=1))
    write_json(WORK/'initial_curation.json',state)
    config=load_config(ROOT/'configs/nuscenes_scene-0063.yaml')
    final,selected=await repair_shapes(WORK/'initial_generation.json',state,config,WORK/'attempts',
        renderer_python='/home/wmj/miniconda3/envs/sim/bin/python',gpus=[0,1,2,3,5])
    r=selected['assets'][0]
    assert r['obj_id']==OID and r['real_images']==[] and final['counts']['queries']==1
    assert r['stages']['H']['real_count']==0
    write_json(WORK/'verified.json',dict(counts=final['counts'],original_real_count=len(row['real_images']),
        retry_real_count=0,production_counted=False,uid=OID,selected_raw_path=r['raw_path'],
        final_audit=final['assets'][OID].get('entry',{}).get('quality',{}).get('audit'),
        limitation='This observed malformed truck already used zero real images; tests separately exercise removal of nonempty references.'))
    print(final['counts'],flush=True)

asyncio.run(main())
