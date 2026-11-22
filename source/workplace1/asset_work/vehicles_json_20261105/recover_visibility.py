"""Recheck exact targets excluded solely by native visibility, under shared GPU leases."""
import asyncio,json,shutil,time
from pathlib import Path
import run
ROOT=run.ROOT
async def main():
 inventory=run.read_json(ROOT/'inventory.json');audit=run.read_json(ROOT/'missing_source_audit/summary.json')
 byid={r['obj_id']:r for r in inventory['assets']};candidates=[]
 for rec in audit['targets']:
  views=[c for c in rec['candidates'] if c['failed_filters']==['low_native_visibility']]
  if not views:continue
  a=dict(byid[rec['obj_id']],candidates=views,status='source_candidates_ready',recovery_reason='Per-camera visual review supersedes global visibility gate',eligible_observations=len(views))
  candidates.append(a)
 run.write_json(ROOT/'missing_source_audit/recovery_plan.json',dict(updated=time.time(),targets=[a['obj_id'] for a in candidates],protocol='Same requested identity, complete projected box, unchanged pixel and visual quality requirements'))
 client=run.AsyncOpenAIClient(run.CONFIG,None)
 try:
  await client.check_endpoints()
  await run.prepare_group(client,candidates)
 finally:await client.close()
 if not (ROOT/'inventory_initial.json').exists():shutil.copyfile(ROOT/'inventory.json',ROOT/'inventory_initial.json')
 recovered=[]
 for a in candidates:
  folder=ROOT/'sources'/a['obj_id']
  if not (folder/'ready.json').exists():continue
  previous=byid[a['obj_id']];previous.update(a);run.write_json(folder/'candidates.json',a);recovered.append(a['obj_id'])
 inventory['counts']=dict(run.collections.Counter(r['status'] for r in inventory['assets']));inventory['updated']=time.time();run.write_json(ROOT/'inventory.json',inventory)
 run.write_json(ROOT/'missing_source_audit/recovery_result.json',dict(updated=time.time(),attempted=len(candidates),recovered=recovered,remaining_original_missing=12-len(recovered)))
 print('Recovered',len(recovered),'of',len(candidates),'attempts',flush=True)
if __name__=='__main__':asyncio.run(main())
