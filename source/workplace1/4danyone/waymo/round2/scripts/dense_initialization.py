"""Actual denser visual-hull surfaces, same21 training views and9 timestamps."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from common_initialization import project,sha,save

R=Path(__file__).resolve().parents[1];W=R.parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=['phone_full','pilot_043_twentyfour','batch_0073_twentyfour','batch_0086_twentyfour','walk063_rootlocal24','batch_0084_twentyfour'],required=True);args=p.parse_args()
    label=args.label;base=R if label=='walk063_rootlocal24' else W;data=base/'reconstruction/data'/label
    origin=json.loads((base/'reconstruction/research'/('full-export.json' if label=='phone_full' else f'{label}-export.json')).read_text())
    source=Path(origin['source']);cameras=json.loads((source/'cameras.json').read_text())['cameras']
    train=[v for v in range(24) if v not in (0,8,16)];assert len(cameras)==24
    output=R/'capacity'/label;output.mkdir(parents=True,exist_ok=False)
    shape=(128,192,128);bounds=[(-1.2,1.2),(-.1,2.1),(-1.2,1.2)]
    grid=np.stack(np.meshgrid(*(np.linspace(lo,hi,n) for (lo,hi),n in zip(bounds,shape,strict=True)),indexing='ij'),-1).reshape(-1,3)
    projections={}
    for view in train:
        uv,valid=project(grid,cameras[view]);projections[view]=(uv.astype(np.int32),valid)
    positions=[];colors=[];times=[];checks=[];inputs=[]
    for frame in range(0,121,15):
        images={}
        for view in train:
            path=data/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
            with Image.open(path) as image:images[view]=np.array(image.convert('RGBA'))
            inputs.append({'view':view,'frame':frame,'path':str(path),'sha256':sha(path)})
        votes=np.zeros(len(grid),np.uint8)
        for view in train:
            uv,valid=projections[view];votes[valid]+=(images[view][uv[valid,1],uv[valid,0],3]>=128)
        occupied=(votes>=math.ceil(.95*len(train))).reshape(shape)
        if any((np.take(occupied,0,axis=axis).any() or np.take(occupied,-1,axis=axis).any()) for axis in range(3)):
            raise ValueError('Hull touches initialization boundary; no implicit clipping')
        ids=np.flatnonzero((occupied&~binary_erosion(occupied)).ravel());points=grid[ids]
        if len(points)<50:raise ValueError('Insufficient foreground-consistent geometry')
        rgb=np.zeros((len(points),3));count=np.zeros(len(points),np.int32)
        for view in train:
            uv,valid=project(points,cameras[view]);valid[valid]&=images[view][uv[valid,1],uv[valid,0],3]>=128
            rgb[valid]+=images[view][uv[valid,1],uv[valid,0],:3];count[valid]+=1
        if count.min()==0:raise ValueError('Unobserved initialization color')
        positions.append(points.astype(np.float32));colors.append((rgb/count[:,None]).astype(np.float32));times.append(np.full(len(points),frame/120,np.float32))
        checks.append({'frame':frame,'points':len(points)});print(json.dumps(checks[-1]),flush=True)
    xyz=np.concatenate(positions);n=len(xyz);original_count=len(np.load(data/'init.npz')['positions'])
    if not 2.5<=n/original_count<=6 or not 35000<=n<=100000:raise ValueError(f'Unexpected dense initialization count:{n}')
    init=output/'init.npz'
    np.savez_compressed(init,positions=xyz,colors=np.concatenate(colors),velocities=np.zeros_like(xyz),times=np.concatenate(times),
        durations=np.full(n,.125,np.float32),frame_start=0,frame_end=121,time_denominator=120,keyframe_step=15,
        mode='dense128x192x128_training21_visual_hull_nine_times_zero_velocity')
    arrays=np.load(init);assert all(np.isfinite(arrays[k]).all() for k in ('positions','colors','times','velocities','durations'))
    save(output/'manifest.json',{'state':'dense_training_only_surface_ready','label':label,'points':n,'original_points':original_count,
        'ratio':n/original_count,'grid_shape':shape,'grid_bounds':bounds,'train_views':train,'excluded_views':[0,8,16],
        'keyframe_checks':checks,'inputs':inputs,'init':str(init),'sha256':sha(init),'base_init_sha256':sha(data/'init.npz'),
        'source_cameras_sha256':sha(source/'cameras.json'),'normalization':'Keep original data and original points3D for unchanged camera normalization; only Gaussian initialization surface density changes.',
        'scope':'Capacity/initialization-density control. New points are real finer-grid hull surface samples, not duplicate points or a raised unused cap. Initial geometry and nearest-neighbor scales also change, so not an identical-initial-render capacity-only test.'})


if __name__=='__main__':main()
