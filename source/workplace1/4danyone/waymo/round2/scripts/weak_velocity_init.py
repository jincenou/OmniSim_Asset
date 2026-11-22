"""R07 body-motion velocity prior; never a claim of cloth/material correspondences."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.spatial import cKDTree

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'moving61_runtime'))
from root_local_geometry import separate_horizontal_root

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import torch
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry
    from fdanyone.assets import resolve_regressor
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in (4,5,6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();assert str(torch.cuda.get_device_properties(0).uuid)==expected
    source=R/'experiments/moving_R07_short61_24_v1'
    root=R/'diagnostics/weak_velocity';root.mkdir(parents=True,exist_ok=False)
    motion_dir=(source/'preflight/motion').resolve()
    motion=MotionResult.load(motion_dir)
    runtime=R/'moving61_runtime'
    geometry=_body_geometry(motion,resolve_regressor(None,model_dir=runtime/'models'),runtime/'third_party/GVHMR','cuda:0')
    local,maps,checks=separate_horizontal_root(geometry)
    original=np.load(source/'preflight/root_motion.npz')
    for key in maps:
        assert np.allclose(maps[key],original[key],atol=1e-7),key
    vertices=np.asarray(local.vertices_world,dtype=np.float64)
    assert vertices.shape==(61,6890,3)
    times=np.arange(61)/60
    derivatives=np.gradient(vertices,times,axis=0,edge_order=2)
    base=R/'reconstruction/data/r07_short61_24/init.npz'
    initial=np.load(base);values={k:initial[k] for k in initial.files}
    assert not values['velocities'].any()
    frames=np.rint(values['times']*60).astype(int)
    assert set(frames)=={0,15,30,45,60}
    velocities=np.zeros_like(values['velocities']);nearest=np.empty(len(frames));influence=np.empty(len(frames))
    for frame in np.unique(frames):
        select=frames==frame
        distance,index=cKDTree(vertices[frame]).query(values['positions'][select],k=4)
        weights=1/(distance+.02)**2;weights/=weights.sum(1,keepdims=True)
        # Distance is in estimated canonical model coordinates, not verified meters.
        confidence=np.exp(-(distance[:,0]/.10)**2)
        confidence[distance[:,0]>.15]=0
        velocities[select]=(derivatives[frame,index]*weights[:,:,None]).sum(1)*confidence[:,None]
        nearest[select]=distance[:,0];influence[select]=confidence
    assert np.isfinite(velocities).all() and (np.linalg.norm(velocities,axis=1)>0).sum()>100
    values['velocities']=velocities
    path=root/'body_velocity_init.npz';np.savez_compressed(path,**values)
    restored=np.load(path)
    changed=[k for k in initial.files if not np.array_equal(initial[k],restored[k])]
    assert changed==['velocities'],changed
    np.savez_compressed(root/'assignment.npz',nearest_distance=nearest,influence=influence,initial_frame=frames)
    report={'state':'weak_velocity_prepared_training_pending','source':str(source),'source_motion':str(motion_dir),
        'motion_sha256':sha(motion_dir/'motion.safetensors'),'base_init':str(base),'base_sha256':sha(base),
        'new_init':str(path),'new_sha256':sha(path),'changed_npz_fields':changed,'points':len(frames),
        'physical_gpu':gpu,'cuda_uuid':expected,'pid':os.getpid(),'geometry_checks':checks,
        'root_maps_verified_against_original':True,'nearest_distance_quantiles':np.quantile(nearest,[0,.5,.9,.99,1]).tolist(),
        'velocity_norm_quantiles':np.quantile(np.linalg.norm(velocities,axis=1),[0,.5,.9,.99,1]).tolist(),
        'nonzero_velocity_points':int((influence>0).sum()),'zero_velocity_points':int((influence==0).sum()),
        'method':'Derivative of corresponding SMPL body vertices in root-local coordinates, normalized t=[0,1]; inverse-distance blend of4 same-time nearby vertices, exp(-(distance/.10)^2), zero influence beyond.15 estimated model units.',
        'unchanged':'All NPZ fields except velocities; point positions/colors/time/duration/order/count, RGBA, cameras and normalization.',
        'limitations':['Weak body-motion prior, not real cloth/bag material tracking. Farther points retain explicit zero-velocity initialization.',
            'Estimated GVHMR motion can be wrong or noisy; coordinate units are not verified meters.',
            'All-frame motion is generation conditioning already; no held-out image pixels used for initialization.']}
    (root/'initialization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'state':report['state'],'points':len(frames),'nonzero':report['nonzero_velocity_points']}),flush=True)

if __name__=='__main__':main()
