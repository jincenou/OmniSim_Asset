"""Concrete CPU checks of export/recovery consistency and frozen color state."""
import json,hashlib,random
from pathlib import Path
import numpy as np,torch
from plyfile import PlyData
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;OLD=ROOT/'optimization_boundary_20261018'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b
def ply(p):return PlyData.read(p)['vertex'].data
def verify_recovery(run,it,state):
    array=ply(run/'point_cloud'/f'iteration_{it}'/'point_cloud.ply');values=state['gaussians']
    mapping={'_xyz':['x','y','z'],'_scaling':['scale_0','scale_1','scale_2'],'_rotation':['rot_0','rot_1','rot_2','rot_3'],'_opacity':['opacity'],'_features_dc':[f'f_dc_{i}' for i in range(3)],'_features_rest':sorted([k for k in array.dtype.names if k.startswith('f_rest_')],key=lambda x:int(x.split('_')[-1]))}
    for name,fields in mapping.items():
        actual=np.stack([array[f] for f in fields],1);expected=values[name].cpu().numpy()
        if name.startswith('_features'):expected=expected.transpose(0,2,1).reshape(len(actual),-1)
        assert np.array_equal(actual,expected),(run.name,it,name)
    d=torch.load(run/'deform'/f'iteration_{it}'/'deform.pth',map_location='cpu',weights_only=False);equal(d[0],state['hash']);equal(d[1],state['deform'])
    return dict(run=run.name,snapshot=it,points=len(array),gaussian_ply_exact=True,hash_deform_exact=True)
def main():
    recovery=[];increments=[];assets=[]
    for label in ['N01','N07']:
        prior=None
        for stage,it in [('33k',33000),('alpha',48000),('A',53000)]:
            run=NEW/'runs'/f'{label}_{stage}';s=torch.load(run/'recovery.pt',map_location='cpu',weights_only=False);recovery.append(verify_recovery(run,it,s))
            if prior:
                budget=15000 if stage=='alpha' else 5000
                for key in ['gaussian_optimizer','deform_optimizer']:
                    before,after=prior[key],s[key];assert [g['name'] for g in before['param_groups']]==[g['name'] for g in after['param_groups']]
                    for g,h in zip(before['param_groups'],after['param_groups']):
                        for i,j in zip(g['params'],h['params']):assert float(after['state'][j]['step'])-float(before['state'][i]['step'])==budget
                increments.append(dict(label=label,stage=stage,all_optimizer_step_increments=budget,fixed_point_count=prior['gaussians']['_xyz'].shape==s['gaussians']['_xyz'].shape))
            prior=s
        del prior,s
    run=NEW/'runs/N04_Bmatch';s=torch.load(run/'recovery.pt',map_location='cpu',weights_only=False);recovery.append(verify_recovery(run,53000,s));del s
    for label in ['N01','N07']:
        run=NEW/'runs'/f'{label}_fused'
        if (run/'complete.json').exists():
            s=torch.load(run/'recovery.pt',map_location='cpu',weights_only=False);recovery.append(verify_recovery(run,33000,s));del s
    source=OLD/'runs/primary/N11_A';base=ply(source/'point_cloud/iteration_53000/point_cloud.ply');base_d=torch.load(source/'deform/iteration_53000/deform.pth',map_location='cpu',weights_only=False);frozen=[f for f in base.dtype.names if not f.startswith('f_dc_') and not f.startswith('f_rest_')];colors=[]
    for arm in ['C1','C2']:
        run=NEW/'runs'/f'N11_{arm}';snapshots=[]
        for epoch in [0,1,5,10,20]:
            a=ply(run/'point_cloud'/f'iteration_{epoch}/point_cloud.ply');assert len(a)==len(base)
            for field in frozen:assert np.array_equal(a[field],base[field]),(arm,epoch,field)
            d=torch.load(run/'deform'/f'iteration_{epoch}/deform.pth',map_location='cpu',weights_only=False);equal(d,base_d)
            state=torch.load(run/'recoveries'/f'color_pass_{epoch}.pt',map_location='cpu',weights_only=False)
            for name,prefix in [('features_dc','f_dc_'),('features_rest','f_rest_')]:
                fields=sorted([f for f in a.dtype.names if f.startswith(prefix)],key=lambda x:int(x.split('_')[-1]));assert np.array_equal(np.stack([a[f] for f in fields],1),state[name].cpu().numpy().transpose(0,2,1).reshape(len(a),-1))
            snapshots.append(dict(pass_number=epoch,color_updates=state['color_updates'],base_grid4d_updates=state['base_grid4d_updates'],frozen_geometry_opacity_point_order_exact=True,hash_deform_exact=True,recovery_SH_exact=True))
        alpha=[]
        for view in [0,8,16]:
            with np.load(OLD/'cache/raw/primary/N11_A'/f'view{view:02d}.npz') as b,np.load(NEW/'cache/raw'/f'N11_{arm}_20'/f'view{view:02d}.npz') as a:
                error=float(np.abs(a['alpha']-b['alpha']).max());assert error<1e-5;alpha.append(dict(view=view,frames=len(a['alpha']),max_abs_error=error))
        colors.append(dict(arm=arm,snapshots=snapshots,full_timeline_alpha_checks=alpha))
    # Reconstruct the deterministic private shuffle state; identify provenance honestly.
    sampler=random.Random(42);states={'0':sampler.getstate()};orders={}
    for epoch in range(1,21):
        order=list(range(168));sampler.shuffle(order);orders[str(epoch)]=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        if epoch in [1,5,10,20]:states[str(epoch)]=sampler.getstate()
    save(NEW/'runs/N11_C1/recoveries/sampler_states.json',dict(state_source='Reconstructed from frozen Python Random(42) and exact168-index shuffle code; not claimed runtime captured',states=states,orders_sha256=orders))
    for run in sorted((NEW/'runs').iterdir()):
        if not (run/'complete.json').exists():continue
        snapshots=[]
        for p in sorted((run/'snapshots').glob('*.json')):
            snapshot=read(p)
            for path,digest in snapshot['files'].items():assert sha(run/path)==digest,(run.name,path)
            validation=run/'validation'/p.name
            snapshots.append(dict(metadata=str(p.relative_to(NEW)),files=snapshot['files'],replay_validation=str(validation.relative_to(NEW)) if validation.exists() else None))
        state_paths=([run/'recovery.pt'] if (run/'recovery.pt').exists() else [])+sorted((run/'recoveries').glob('*'))
        states=[dict(path=str(p.relative_to(NEW)),bytes=p.stat().st_size,sha256=sha(p)) for p in state_paths if p.is_file()]
        assets.append(dict(root=str(run.relative_to(NEW)),config=str((run/'config.json').relative_to(NEW)),camera_data=str((run/'asset_camera_data.json').relative_to(NEW)),snapshots=snapshots,recovery_components=states,complete=read(run/'complete.json')))
    solver_meta=read(NEW/'runs/N11_C2/recoveries/solver_state.json')
    assert sha(NEW/'runs/N11_C2/recoveries'/solver_meta['path'])==solver_meta['sha256']
    save(NEW/'research/recovery_audit.json',dict(state='passed_for_all_eleven_complete_runs',full_recoveries=recovery,optimizer_increments=increments,color_frozen_checks=colors,color_solver_state=solver_meta,limits='C1 sampler state reconstructed from deterministic schedule; C2 solver matrix reconstituted and saved from the same fixed geometry/training pool/numeric recipe, not claimed byte-identical to its original temporary GPU reduction. Asset-only replay needs neither sampler nor solver state and does not read training observations.'))
    save(NEW/'ASSETS.json',dict(state='inventory',assets=assets,environment='/mnt/ojc/miniconda3/envs/grid4d',runtime=str(ROOT/'source'),replay_script=str(ROOT/'scripts/replay_asset.py'),dynamic_contract='Canonical Gaussian PLY + matched hash/deform + config + cameras + native runtime. Not novel action driving; geometry units are normalized.'))
    print('Recovery and frozen state checks passed',len(assets),flush=True)
if __name__=='__main__':main()
