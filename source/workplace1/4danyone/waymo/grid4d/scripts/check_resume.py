"""Compare uninterrupted and explicitly interrupted/restarted technical smoke."""
import json
from pathlib import Path
import numpy as np
from grid_common import ROOT, save, stamp


def main():
    a=ROOT/'runs/N04_smoke_full_v2';b=ROOT/'runs/N04_smoke_resume_v2'
    assert (a/'complete.json').exists() and (b/'complete.json').exists()
    first=[json.loads(s) for s in (a/'trace.jsonl').read_text().splitlines()]
    second=[json.loads(s) for s in (b/'trace.jsonl').read_text().splitlines()]
    assert len(first)==120 and len(second)==60
    first=first[60:]
    assert [(x['iteration'],x['view'],x['frame'],x['points']) for x in first]==[(x['iteration'],x['view'],x['frame'],x['points']) for x in second]
    loss=max(abs(x['loss']-y['loss']) for x,y in zip(first,second));assert loss<1e-4,loss
    x=np.load(a/'snapshots/120_reference.npz');y=np.load(b/'snapshots/120_reference.npz')
    delta=max(float(np.abs(x[k]-y[k]).max()) for k in x.files);assert delta<1e-4,delta
    save(ROOT/'research/resume_probe.json',dict(state='passed',at=stamp(),iterations=120,
        interruption_after_iteration=60,sampling_and_point_count_sequence_identical=True,
        max_loss_difference=loss,max_reference_render_difference=delta,tolerance=1e-4,
        purpose='Same exact prefix60 checkpoint: in-memory continuation vs independent reload, same GPU. Complete optimizer/RNG/camera stack/densification restore technical test; no quality result',
        earlier_test_limitation='Two independently trained prefixes differed before interruption due to numerical nondeterminism; preserved but not used to test restore equivalence'))
    print('Resume comparison passed',loss,delta,flush=True)


if __name__=='__main__':main()
