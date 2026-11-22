"""Audit the immutable completed cases and distinguish pending pilot work."""
import hashlib,json,re
import numpy as np
from run_queue import ROOT,save,stamp
BASE=ROOT.parent/'random8/r20261015_113332'
MAIN=['N01','N02','N03','N04','N06','N07','N08']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    checked=[]
    for label in MAIN+['N11']:
        p=ROOT/'results'/label/'results.json'
        if not p.exists():
            assert label=='N11'
            continue
        result=json.loads(p.read_text())
        data=json.loads((BASE/'four_dg/data'/label/'fdanyone.json').read_text())
        stages=[]
        for s in [1,2]:
            d=ROOT/'runs'/label/f'stage{s}'
            trace=np.load(d/'sampling.npy')
            comp=json.loads((d/'complete.json').read_text())
            inp=json.loads((d/'input.json').read_text())
            assert set(trace[:,1])==set(data['train_views']) and not set(trace[:,1])&{0,8,16}
            assert set(trace[:,2])==set(range(data['frame_count']))
            assert comp['updates_this_stage']==(26999 if s==1 else 7999)
            assert digest(d/'asset.pt')==comp['asset_sha256']
            assert digest(BASE/'four_dg/data'/label/data['canonical_npz'])==inp['initialization_sha256']
            replay=result['replay'][f'cem_stage{s}']
            assert len(replay)==9 and max(v['max_error'] for v in replay)<1e-5
            stages.append({'stage':s,'samples':len(trace),'unique_train_views':len(set(trace[:,1])),
                'unique_frames':len(set(trace[:,2])),'asset_sha256':comp['asset_sha256'],
                'replay_max_error':max(v['max_error'] for v in replay)})
        for method,rows in result['samples'].items():
            assert len(rows)==39 and set(v['view'] for v in rows)=={0,8,16}
        assert all(v<1e-4 for v in result['baseline_metric_max_absolute_error'].values())
        for name,info in result['videos'].items():
            assert info['frames']==data['frame_count']
            assert digest(p.parent/name)==info['sha256']
        checked.append({'label':label,'state':'passed','stages':stages,'complete_videos':len(result['videos'])})
    links=re.findall(r'\]\(([^)]+)\)',(ROOT/'REPORT.md').read_text())
    local=[p for p in links if not p.startswith(('https://','http://','#'))]
    missing=[p for p in local if not (ROOT/p.split('#')[0]).exists()]
    assert not missing,missing
    diagnostic=[]
    for label in ['N04','N08']:
        diag=ROOT/'diagnostics/opacity'
        result=json.loads((diag/'results'/label/'results.json').read_text())
        records=[json.loads((diag/'runs'/label/c/'complete.json').read_text()) for c in ['control','mask']]
        assert records[0]['frozen_parameters_sha256']==records[1]['frozen_parameters_sha256']
        assert records[0]['sampling_sha256']==records[1]['sampling_sha256']
        source=json.loads((ROOT/'runs'/label/'stage2/complete.json').read_text())['asset_sha256']
        for condition,record in zip(['control','mask'],records):
            run=diag/'runs'/label/condition
            assert record['actual_extra_updates']==1000 and record['source_asset_sha256']==source
            assert digest(run/'asset.pt')==record['asset_sha256']
            assert digest(run/'sampling.npy')==record['sampling_sha256']
            trace=np.load(run/'sampling.npy')
            assert len(trace)==1000 and not set(trace[:,0])&{0,8,16}
        assert all(v<1e-4 for v in result['original_metric_max_error'].values())
        assert all(len(rows)==39 for rows in result['samples'].values())
        assert all(len(rows)==9 and max(r['max_error'] for r in rows)<1e-5 for rows in result['replay'].values())
        assert result['video']['frames']==61
        assert digest(diag/'results'/label/'comparison.mp4')==result['video']['sha256']
        diagnostic.append({'label':label,'state':'passed','equal_updates':1000,'frozen_parameters_equal':True,
            'equal_sampling':True,'baseline_recompute_max_error':max(result['original_metric_max_error'].values())})
    save(ROOT/'research/delivery_audit.json',dict(updated_at=stamp(),main7_state='passed',
        whole_pilot_state='complete_numerical_delivery' if len(checked)==8 else 'incomplete_N11_recovery_running',
        cases=checked,opacity_diagnostic=diagnostic,local_report_links_checked=len(local),missing_report_links=missing,
        scope='Sampling, actual updates, checkpoint hash, common t0 initialization, independent replay records, heldout pairs, video hashes and report paths. Does not prove real reconstruction quality.'))
    print(json.dumps({'audited_cases':len(checked),'report_links':len(local),'state':'passed'}))

if __name__=='__main__':main()
