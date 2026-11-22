"""Inspect every preserved complete optimizer recovery, not just file existence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
NEW=Path(__file__).resolve().parents[1]


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    plan=json.loads((NEW/'PLAN.json').read_text());rows=[]
    for label,item in plan['identities'].items():
        sampler=torch.load(item['sampling']['primary']['state_path'],map_location='cpu',weights_only=False)
        sequence=json.loads(Path(item['sampling']['primary']['path']).read_text());rates=json.loads(Path(item['lr']['path']).read_text())
        for arm in ['A','U','B']:
            run=NEW/'runs/primary'/f'{label}_{arm}'
            for step in [1000,3000,5000]:
                it=48000+step;recovery=run/'recoveries'/f'{it}.pt';snapshot=json.loads((run/'snapshots'/f'{it}.json').read_text());assert sha(recovery)==snapshot['files'][str(recovery.relative_to(run))]
                state=torch.load(recovery,map_location='cpu',weights_only=False)
                assert state['stage_update']==step and state['global_actual_updates']==47999+step and len(state['gaussians']['_xyz'])==item['points']
                assert state['actual_samples']==[r['uid'] for r in sequence[:step]];same(state['stack'],sampler['checkpoints'][step]['stack']);same(state['rng']['python'],sampler['checkpoints'][step]['python_rng'])
                assert set(state['rng'])=={'python','numpy','torch','cuda'} and len(state['rng']['cuda'])==1
                mappings={}
                for optimizer in ['gaussian_optimizer','deform_optimizer']:
                    actual=state[optimizer];initial=item['optimizer_groups'][optimizer];assert [g['name'] for g in actual['param_groups']]==[g['name'] for g in initial]
                    for group,old in zip(actual['param_groups'],initial):
                        assert group['lr']==rates[step-1][group['name']] and group['eps']==old['eps'] and list(group['betas'])==old['betas']
                        assert group['params']==[r['id'] for r in old['state']]
                        for pid,prior in zip(group['params'],old['state']):
                            current=actual['state'][pid];assert float(current['step'])==prior['step']+step and list(current['exp_avg'].shape)==prior['shape']
                            assert torch.isfinite(current['exp_avg']).all() and torch.isfinite(current['exp_avg_sq']).all()
                        mappings[group['name']]=[float(actual['state'][pid]['step']) for pid in group['params']]
                for name,tensor in state['gaussians'].items():assert torch.isfinite(tensor).all()
                for component in ['hash','deform']:
                    for tensor in state[component].values():
                        if torch.is_tensor(tensor):assert torch.isfinite(tensor).all()
                ply=PlyData.read(str(run/'point_cloud'/f'iteration_{it}'/'point_cloud.ply'))['vertex']
                for index,name in enumerate(['x','y','z']):assert np.array_equal(ply[name],state['gaussians']['_xyz'][:,index].numpy())
                assert np.array_equal(ply['opacity'],state['gaussians']['_opacity'][:,0].numpy())
                for index in range(3):assert np.array_equal(ply[f'scale_{index}'],state['gaussians']['_scaling'][:,index].numpy())
                for index in range(4):assert np.array_equal(ply[f'rot_{index}'],state['gaussians']['_rotation'][:,index].numpy())
                for index in range(3):assert np.array_equal(ply[f'f_dc_{index}'],state['gaussians']['_features_dc'][:,0,index].numpy())
                rest=state['gaussians']['_features_rest'].transpose(1,2).flatten(start_dim=1).numpy()
                for index in range(rest.shape[1]):assert np.array_equal(ply[f'f_rest_{index}'],rest[:,index])
                weights=torch.load(run/'deform'/f'iteration_{it}'/'deform.pth',map_location='cpu',weights_only=False)
                assert isinstance(weights,tuple) and len(weights)==2
                same(weights[0],state['hash']);same(weights[1],state['deform'])
                rows.append(dict(label=label,arm=arm,iteration=it,actual_updates=47999+step,adam_steps=mappings,point_count=item['points'],gaussian_ply_exact=True,rng_stack_and_sampling_exact=True,optimizer_tensor_states_finite=True,recovery_sha256=sha(recovery),hash_and_deform_asset_exact=True))
                del state,weights
    (NEW/'research/recovery_audit.json').write_text(json.dumps(dict(state='passed',snapshots=rows),indent=2)+'\n');print('All18 complete recovery states inspected',flush=True)


if __name__=='__main__':main()
