"""Use the five other authorized GPUs for five fixed, independent full CEM cases."""
import asyncio,json,os
from run_queue import ROOT,query_gpus,case,save,stamp

async def main():
    assert json.loads((ROOT/'research/cec_probe.json').read_text())['run_id']=='installed'
    assert json.loads((ROOT/'research/extra_inputs.json').read_text())['state']=='passed'
    labels=['N01','N02','N03','N06','N07'];physical=[1,4,5,6,7]
    inventory={g['physical_gpu']:g for g in await query_gpus()}
    gpus=[inventory[i] for i in physical]
    assert all(g['free_mib']>16000 for g in gpus)
    assignments=dict(zip(labels,gpus));state={'state':'five_additional_cases_running','started_at':stamp(),
        'pid':os.getpid(),'assignments':assignments,'selection':'Remaining completed original Random8 baselines'}
    save(ROOT/'EXTRA_STATUS.json',state)
    results=await asyncio.gather(*map(lambda pair:case(*pair),assignments.items()),return_exceptions=True)
    state.update(state='evaluated_visual_review_pending' if not any(isinstance(r,Exception) for r in results) else 'failed_partial',
        ended_at=stamp(),results=[str(r) if isinstance(r,Exception) else r for r in results])
    save(ROOT/'EXTRA_STATUS.json',state)
    assert state['state']!='failed_partial',state['results']

if __name__=='__main__':asyncio.run(main())
