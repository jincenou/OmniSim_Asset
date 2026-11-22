"""Fill releasedGPU lanes after confirmed failures,without exceeding20 possible successes."""
import asyncio,json,os,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def hard_failure(key,job):
 stage=job.get('stage');p=R/'logs'/f'{key}_{stage}.log';log=p.read_text() if p.exists() else ''
 review=R/'research/reviewed_rejections.json'
 if review.exists():
  row=read(review).get(key)
  if row and row['stage']==stage and row['log_sha256']==__import__('hashlib').sha256(p.read_bytes()).hexdigest():return True
 return (stage=='interpolate' and 'Exposure interval lacks coverage beyond frozen1ms timing tolerance' in log) or (stage=='composite' and 'Common frontend foreground gate failed' in log) or (stage=='conditioning' and 'Root-local branch still needs excessive camera radius' in log)
async def main():
 admitted=read(R/'reserve_admitted.json')['ids'] if (R/'reserve_admitted.json').exists() else [];history=read(R/'research/reserve_history.json') if (R/'research/reserve_history.json').exists() else []
 while True:
  p=read(R/'STATUS.json');r=read(R/'RESERVE_STATUS.json') if (R/'RESERVE_STATUS.json').exists() else {'jobs':{},'state':'idle'};jobs={**p['jobs'],**r['jobs']};complete=sum(j.get('state')=='complete' for j in jobs.values());unknown=[k for k,j in jobs.items() if j.get('state')=='failed' and not hard_failure(k,j)]
  if unknown:
   (R/'reserve_decision.json').write_text(json.dumps({'state':'requires_infrastructure_review','jobs':unknown,'complete':complete},indent=2));break
  primary_possible=20-sum(j.get('state')=='failed' for j in p['jobs'].values());reserve_complete=sum(j.get('state')=='complete' for j in r['jobs'].values());slots=20-primary_possible-reserve_complete
  candidates=[x for x in read(R/'reserve_manifest.json')['candidates'] if x['sample_id'] not in jobs and x['sample_id'] not in admitted]
  if complete>=20:
   (R/'reserve_decision.json').write_text(json.dumps({'state':'target_reached','complete':complete,'attempts':len(jobs)},indent=2));break
  if slots<=0 or not candidates:
   if p['state']!='running' and r['state']!='running':
    (R/'reserve_decision.json').write_text(json.dumps({'state':'bounded_pool_exhausted_or_all_slots_terminal','complete':complete,'attempts':len(jobs)},indent=2));break
   await asyncio.sleep(20);continue
  chosen=candidates[:slots];ids=[x['sample_id'] for x in chosen];admitted+=ids;(R/'reserve_admitted.json').write_text(json.dumps({'ids':admitted,'history':history,'primary_possible_at_admission':primary_possible,'rule':'ordered hardfailure replacement,20maximum potential successes'},indent=2));env=os.environ.copy();env['FD_SCENE20_BATCH_IDS']=','.join(ids);env['FD_SCENE20_STATUS']='RESERVE_STATUS.json'
  with (R/'logs'/f'reserve_batch{len(history)+1}.log').open('a') as log:
   proc=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/4danyone/bin/python',str(R/'scripts/scheduler.py'),env=env,stdout=log,stderr=log);rc=await proc.wait();history.append({'ids':ids,'exit_code':rc,'at':time.time()});(R/'research/reserve_history.json').write_text(json.dumps(history,indent=2));assert rc==0
 (R/'reserve_pending.flag').unlink()
if __name__=='__main__':asyncio.run(main())
