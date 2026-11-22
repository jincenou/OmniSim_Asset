"""Calibrated full-time export and hull initialization excluding held-out views."""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

R=Path(__file__).resolve().parents[1]
W=Path('/mnt/ojc/workplace1/4danyone/waymo')
LABEL=os.environ['FDANYONE_ASSET_NAME']
assert LABEL.replace('_','').isalnum()
SOURCE=Path(os.environ['FDANYONE_ASSET_SOURCE'])
DEST=R/'data'/LABEL


def project(points,camera):
    w=np.linalg.inv(camera['camera_to_world']);p=points@w[:3,:3].T+w[:3,3]
    K=np.array(camera['K']);K[:2]*=.5;q=p@K.T
    uv=np.rint(q[:,:2]/q[:,2:]).astype(int)
    valid=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<352)&(uv[:,1]>=0)&(uv[:,1]<640)
    return uv,valid


def main():
    if (DEST/'init.npz').exists():raise FileExistsError(DEST/'init.npz')
    cams=json.loads((SOURCE/'cameras.json').read_text())['cameras']
    n=len(cams);assert n==24
    assert json.loads((SOURCE/'cameras.json').read_text())['world_frame']['name']=='root_local_human_world'
    assert json.loads((SOURCE/'metadata.json').read_text())['output']['fps']=='25/1'
    test=[0] if n==6 else [0,8,16];train=[i for i in range(n) if i not in test]
    shape=(64,96,64)
    axes=np.meshgrid(np.linspace(-1.2,1.2,64),np.linspace(-.1,2.1,96),np.linspace(-1.2,1.2,64),indexing='ij')
    grid=np.stack(axes,-1).reshape(-1,3);projections=[project(grid,c) for c in cams]
    positions=[];colors=[];times=[];checks=[]
    for frame in range(0,61,15):
        images=[np.array(Image.open(DEST/'images'/f'cam{i:02d}'/f'{frame:06d}.png')) for i in range(n)]
        assert all(im.shape==(640,352,4) for im in images)
        votes=np.zeros(len(grid),np.int16)
        for i in train:
            uv,valid=projections[i];hit=np.zeros(len(grid),bool)
            hit[valid]=images[i][uv[valid,1],uv[valid,0],3]>=128;votes+=hit
        occupied=(votes>=math.ceil(.95*len(train))).reshape(shape)
        if any(face.any() for face in (occupied[0],occupied[-1],occupied[:,0],occupied[:,-1],occupied[:,:,0],occupied[:,:,-1])):raise RuntimeError('Visual hull touches grid boundary; do not silently clip the asset')
        indices=np.flatnonzero((occupied&~binary_erosion(occupied)).ravel())
        if len(indices)<50:raise RuntimeError(f'Insufficient geometry at frame {frame}: {len(indices)}; review grid or generated consistency')
        pts=grid[indices];rgb=np.zeros((len(pts),3));count=np.zeros(len(pts))
        for i in train:
            uv,valid=project(pts,cams[i]);valid[valid]&=images[i][uv[valid,1],uv[valid,0],3]>=128
            rgb[valid]+=images[i][uv[valid,1],uv[valid,0],:3];count[valid]+=1
        assert count.min()>0
        rgb/=count[:,None];positions.append(pts.astype(np.float32));colors.append(rgb.astype(np.float32))
        times.append(np.full(len(pts),frame/60,np.float32));checks.append({'frame':frame,'seconds':frame/25,'surface_points':len(pts)})
    pos=np.concatenate(positions);rgb=np.concatenate(colors)
    np.savez_compressed(DEST/'init.npz',positions=pos,colors=rgb,velocities=np.zeros_like(pos),times=np.concatenate(times),
        durations=np.full(len(pos),.25,np.float32),frame_start=0,frame_end=61,time_denominator=60,keyframe_step=15,
        mode='five_visual_hull_keyframes_zero_initial_velocity')
    sparse=DEST/'sparse/0';sparse.mkdir(parents=True,exist_ok=False);cameras=[];images=[]
    for idx,c in enumerate(cams,1):
        K=np.array(c['K']);K[:2]*=.5
        cameras.append(f'{idx} PINHOLE 352 640 {K[0,0]:.12g} {K[1,1]:.12g} {K[0,2]:.12g} {K[1,2]:.12g}')
        w=np.linalg.inv(c['camera_to_world']);q=Rotation.from_matrix(w[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
        values=' '.join(f'{v:.15g}' for v in np.r_[q,w[:3,3]])
        images.append(f'{idx} {values} {idx} cam{idx-1:02d}/000000.png\n')
    (sparse/'cameras.txt').write_text('\n'.join(cameras)+'\n');(sparse/'images.txt').write_text('\n'.join(images)+'\n')
    lines=[f'{i+1} {p[0]:.8g} {p[1]:.8g} {p[2]:.8g} {int(c[0])} {int(c[1])} {int(c[2])} 0' for i,(p,c) in enumerate(zip(positions[0],colors[0],strict=True))]
    (sparse/'points3D.txt').write_text('\n'.join(lines)+'\n')
    sys.path[:0]=[str(R.parent/'backend/src'),str(R.parent/'backend')]
    from datasets.FreeTime_dataset import FreeTimeParser,FreeTimeDataset
    raw=FreeTimeParser(str(DEST),normalize=False,start_frame=0,end_frame=61)
    parser=FreeTimeParser(str(DEST),normalize=True,start_frame=0,end_frame=61)
    for idx,c in enumerate(cams):
        K=np.array(c['K']);K[:2]*=.5
        assert np.allclose(raw.camtoworlds[idx],c['camera_to_world'],atol=1e-6)
        assert np.allclose(raw.Ks_dict[idx+1],K,atol=1e-6)
    datasets=[FreeTimeDataset(parser,split=split,test_set=test,alpha_mode='transparent') for split in ('train','val')]
    assert [len(d) for d in datasets]==[len(train)*61,len(test)*61]
    for dataset in datasets:
        for index in (0,len(dataset)//2,len(dataset)-1):
            item=dataset[index]
            assert tuple(item['image'].shape)==(640,352,3) and tuple(item['alpha'].shape)==(640,352,1)
            assert np.isclose(item['time'],item['frame_offset']/60)
    result={'status':'full_export_actual_loader_passed','source':str(SOURCE),'asset_label':LABEL,'views':n,'frames':61,
        'dimensions':[352,640],'fps':25,'sample_span_seconds':2.4,'train_views':train,'test_views':test,
        'train_samples':len(datasets[0]),'test_samples':len(datasets[1]),'initialization_points':len(pos),'keyframe_checks':checks,
        'initial_duration_sigma':.25,'initial_velocity':'zero, learned during optimization',
        'K_transform':'first two rows multiplied by 0.5','camera_convention':'OpenCV w2c=inverse(c2w)',
        'backend_normalization_transform':parser.transform.tolist(),'grid_bounds':[[-1.2,1.2],[-.1,2.1],[-1.2,1.2]],
        'scope':'Generated observations only. Held-out cameras excluded from carving and colors. Root-local Y-up estimated coordinates before normalization; real Waymo scale not calibrated.',
        'limitations':['Independent backend, not the unreleased paper implementation','Six-view pilots have fewer constraints than the 24-view baseline']}
    (R/'research'/f'{LABEL}-export.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
