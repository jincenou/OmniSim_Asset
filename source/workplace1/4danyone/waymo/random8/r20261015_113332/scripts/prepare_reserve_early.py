"""Prepare only the next justified preregistered reserve while two main videos run.

No video generation or backend launch. Main scheduler will reuse validated
prefixes; registry coordination prevents duplicate preparation.
"""
import asyncio,json,os,datetime
from pathlib import Path
import controller_v3 as c
from reserve_admission import verify
R=Path(__file__).resolve().parents[1];LABEL='N11';STATE=R/'FRONTEND_REPAIR_STATUS.json';PROOF=R/'samples'/LABEL/'generation/preflight/review.json'
def save():
 value=json.loads(STATE.read_text());jobs={k:v for k,v in c.state['jobs'].items() if k.startswith(LABEL+'/')};current=list(jobs.values())[-1] if jobs else {};row=value['cases'].setdefault(LABEL,{});row.update(state='running',kind='early_preparation_of_next_justified_reserve',job=current,jobs=jobs,pid=os.getpid());value['state']='registered_reserve_preparing';value['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=STATE.with_suffix('.external-writing');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATE)
def finish(status,**extra):
 value=json.loads(STATE.read_text());value['cases'][LABEL].update(state=status,**extra);value['state']='bounded_frontend_repairs_terminal';tmp=STATE.with_suffix('.external-writing');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATE)
async def main():
 row=next(x for x in c.selection['reserve'] if x['sample_id']==LABEL);verify(row,R)
 if PROOF.exists():return
 c.state={'jobs':{},'samples':{},'state':'preparing_next_justified_reserve'};c.persist=save;save();base=R/'samples'/LABEL;src=base/'source'
 try:
  async with c.disk_budget(LABEL+'/early_source',.12):
   await c.stage(LABEL,'camera_audit','hust','raw_camera_audit.py',['--review-id',LABEL,'--manifest',R/'selection_manifest.json'],proof=R/'camera_audit'/LABEL/'audit.json')
   await c.stage(LABEL,'native','front','m02_prepare.py',proof=src/'native.json')
   gpu=await c.choose_gpu(6500);await c.stage(LABEL,'interpolate','front','interpolate.py',gpus=[gpu],minimum=6500,proof=src/'inputs/frame_mapping.json')
   await c.stage(LABEL,'source_prepare','front','source_foreground.py',['prepare'],proof=src/'segmentation/frames/00000.jpg')
   gpu=await c.choose_gpu(7000);await c.stage(LABEL,'source_segment','sam','source_foreground.py',['segment'],gpus=[gpu],minimum=7000,proof=src/'segmentation/target_masks.npz')
   await c.stage(LABEL,'source_composite','front','source_foreground.py',['composite'],proof=src/'research/batch_gate_mask.json');assert json.loads((src/'research/batch_gate_mask.json').read_text())['passed']
   gpu=await c.choose_gpu(15000);await c.stage(LABEL,'motion','front','motion.py',gpus=[gpu],minimum=15000,proof=src/'motion_pair/calibrated_moving/motion.safetensors')
   await c.stage(LABEL,'motion_gate','front','source_foreground.py',['motion_gate'],proof=src/'research/motion_gate.json')
   gpu=await c.choose_gpu(9000);await c.stage(LABEL,'conditioning','front','conditioning.py',gpus=[gpu],minimum=9000,proof=PROOF)
  assert json.loads(PROOF.read_text())['allow_generation'];finish('same_input_repaired_ready_for_first24view_generation',ready_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except Exception as error:
  text='\n'.join(Path(v['log']).read_text(errors='replace')[-5000:] for v in c.state['jobs'].values() if v.get('log'))
  hard=any(x in text for x in ['Common frontend foreground gate failed','Illegal exposure-time continuity','Exposure interval lacks coverage','Annotated person clipped by crop','Global/incam SMPL geometry is not a consistent rigid pair'])
  finish('failed_after_bounded_repair' if hard else 'blocked_infrastructure',error=f'{type(error).__name__}:{error}',reserve_eligible=hard)
if __name__=='__main__':asyncio.run(main())
