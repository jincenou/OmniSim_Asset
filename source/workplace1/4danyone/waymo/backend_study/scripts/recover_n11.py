"""Preserve the failed attempt, then rerun the same frozen N11 recipe once."""
import asyncio, hashlib, json, os
from run_queue import ROOT,query_gpus,case,save,stamp

async def main():
    state_path=ROOT/'RECOVERY_STATUS.json'
    assert not state_path.exists(),'One explicit recovery attempt only'
    check=json.loads((ROOT/'research/promotion_regression.json').read_text())
    source=ROOT/'sources/cem4dgs/scene/v_gaussian_model.py'
    assert check['state']=='passed'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==check['tested_source_sha256']
    receipt=ROOT/'logs/N11_stage1.json'
    prior=json.loads(receipt.read_text())
    assert prior['state']=='failed' and prior['exit_code']==1
    log=ROOT/'logs/N11_stage1.log'
    assert 'new_opacity_duration_center[prune_child_mask]' in log.read_text()
    assert not (ROOT/'runs/N11/stage1/complete.json').exists()
    assert not (ROOT/'results/N11').exists()
    gpu=next(g for g in await query_gpus() if g['physical_gpu']==6)
    assert gpu['free_mib']>16000
    archive=ROOT/'attempts/N11_stage1_failed'
    archive.mkdir(parents=True,exist_ok=False)
    for path in [ROOT/'runs/N11/stage1',log,receipt]:
        path.rename(archive/path.name)
    state={'state':'rerunning_from_seed42','started_at':stamp(),'pid':os.getpid(),
           'gpu':gpu,'failed_attempt':str(archive),'patch_validation':check,
           'note':'No intermediate checkpoint existed. Full 27k+8k rerun, no changed samples, seed, loss or budget.'}
    save(state_path,state)
    try:
        result=await case('N11',gpu)
    except Exception as exc:
        state.update(state='failed_requires_review',error=str(exc),ended_at=stamp())
        save(state_path,state)
        raise
    state.update(state='evaluated_visual_review_pending',result=result,ended_at=stamp())
    save(state_path,state)

if __name__=='__main__':asyncio.run(main())
