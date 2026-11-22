"""Export a three-timestamp COLMAP/RGBA/NPZ interface probe, not a trained asset."""
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
LABEL=os.environ.get('FDANYONE_BRIDGE_LABEL','official_sample')
assert LABEL in ('official_sample','waymo_sample')
SOURCE=Path(os.environ.get('FDANYONE_BRIDGE_SOURCE',str(ROOT.parent/'outputs/turbo')))
DEST=ROOT/'bridge'/LABEL
MANIFEST=ROOT/'bridge_manifest.json' if LABEL=='official_sample' else ROOT/'bridge/waymo_manifest.json'


def project(points,camera):
    w2c=np.linalg.inv(camera['camera_to_world'])
    p=points@w2c[:3,:3].T+w2c[:3,3]
    K=np.array(camera['K']);K[:2]*=.25
    q=p@K.T
    uv=np.rint(q[:,:2]/q[:,2:]).astype(int)
    valid=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<176)&(uv[:,1]>=0)&(uv[:,1]<320)
    return uv,valid


def main():
    cams=json.loads((SOURCE/'cameras.json').read_text())['cameras']
    assert json.loads((SOURCE/'metadata.json').read_text())['output']['fps']=='25/1'
    assert len(cams)==24
    holdout=[0,8,16]
    train=[i for i in range(24) if i not in holdout]
    shape=(64,96,64)
    coordinates=np.meshgrid(np.linspace(-1.2,1.2,shape[0]),np.linspace(-.1,2.1,shape[1]),np.linspace(-1.2,1.2,shape[2]),indexing='ij')
    grid=np.stack(coordinates,-1).reshape(-1,3)
    projections=[project(grid,c) for c in cams]
    positions=[];colors=[];times=[];checks=[]
    for frame in range(3):
        images=[np.array(Image.open(DEST/'images'/f'cam{i:02d}'/f'{frame:06d}.png')) for i in range(24)]
        votes=np.zeros(len(grid),np.int16)
        for i in train:
            uv,valid=projections[i];hit=np.zeros(len(grid),bool)
            hit[valid]=images[i][uv[valid,1],uv[valid,0],3]>=128
            votes+=hit
        threshold=math.ceil(.95*len(train))
        occupied=(votes>=threshold).reshape(shape)
        surface=occupied&~binary_erosion(occupied)
        indices=np.flatnonzero(surface.reshape(-1))
        if len(indices)<50:raise RuntimeError(f'Insufficient foreground-consistent geometry at frame {frame}: {len(indices)}')
        pts=grid[indices]
        rgb=np.zeros((len(pts),3));count=np.zeros(len(pts))
        for i in train:
            uv,valid=project(pts,cams[i]);valid[valid]&=images[i][uv[valid,1],uv[valid,0],3]>=128
            rgb[valid]+=images[i][uv[valid,1],uv[valid,0],:3];count[valid]+=1
        assert count.min()>0
        rgb/=count[:,None]
        positions.append(pts.astype(np.float32));colors.append(rgb.astype(np.float32));times.append(np.full(len(pts),frame/2,np.float32))
        held=[]
        for i in holdout:
            uv,valid=project(pts,cams[i]);inside=np.zeros(len(pts),bool);inside[valid]=images[i][uv[valid,1],uv[valid,0],3]>=128
            held.append({'view':i,'surface_points_inside_predicted_alpha_fraction':float(inside.mean())})
        checks.append({'bridge_frame':frame,'source_frame':frame*60,'seconds':frame*2.4,'occupied_voxels':int(occupied.sum()),'surface_points':len(pts),'holdout_projection_checks':held})
    pos=np.concatenate(positions);rgb=np.concatenate(colors);t=np.concatenate(times)
    np.savez_compressed(DEST/'init.npz',positions=pos,colors=rgb,velocities=np.zeros_like(pos),times=t,durations=np.full(len(pos),.5,np.float32),frame_start=0,frame_end=3,time_denominator=2,keyframe_step=1,mode='explicit_sampled_visual_hull_zero_velocity')
    sparse=DEST/'sparse/0';sparse.mkdir(parents=True,exist_ok=True)
    cameras=[];images=[]
    for idx,c in enumerate(cams,1):
        K=np.array(c['K']);K[:2]*=.25
        cameras.append(f'{idx} PINHOLE 176 320 {K[0,0]:.12g} {K[1,1]:.12g} {K[0,2]:.12g} {K[1,2]:.12g}')
        w=np.linalg.inv(c['camera_to_world']);q=Rotation.from_matrix(w[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
        vals=' '.join(f'{v:.15g}' for v in np.r_[q,w[:3,3]])
        images.append(f'{idx} {vals} {idx} cam{idx-1:02d}/000000.png\n')
    (sparse/'cameras.txt').write_text('\n'.join(cameras)+'\n')
    (sparse/'images.txt').write_text('\n'.join(images)+'\n')
    lines=[f"{i+1} {p[0]:.8g} {p[1]:.8g} {p[2]:.8g} {int(c[0])} {int(c[1])} {int(c[2])} 0" for i,(p,c) in enumerate(zip(positions[0],colors[0],strict=True))]
    (sparse/'points3D.txt').write_text('\n'.join(lines)+'\n')
    manifest={'status':'exported_loader_check_pending','backend_repository':'https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla',
        'backend_commit':'1f160e290b00d88e097739ecc26d93500d6c55e8','source':str(SOURCE),
        'views':24,'dimensions':[176,320],'frames':3,'source_frame_indices':[0,60,120],'seconds':[0,2.4,4.8],'normalized_time':[0,.5,1],
        'world':'4DAnyone canonical meters Y-up retained','camera':'OpenCV x-right y-down z-forward; w2c=inverse(c2w)',
        'K_transform':'first two rows multiplied by 0.25','distortion':'none in generated virtual camera',
        'source_world_to_zup':[[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],
        'zup_scope':'Axis orientation only; ground offset, heading and Waymo world alignment remain unresolved.',
        'colmap_point_error':'Zero is a required-format placeholder, not measured SfM reprojection error.',
        'foreground':'Per-generated-view BiRefNet soft alpha, straight RGBA, not source mask and not black-pixel masking',
        'train_views':train,'test_views':holdout,'carving':{'grid_shape':shape,'threshold':'at least 20 of 21 training-view masks >=128','scope':'consensus visual hull surface, not metric ground truth'},
        'initialization':{'point_count':len(pos),'velocities':'zeros in meters per exported frame; backend multiplies by time denominator 2',
          'times':'normalized 0,0.5,1','duration_sigma':.5,'motion_correspondences':'not estimated; no claim of learned motion'},
        'frame_checks':checks,'limitations':['Three sampled timestamps only; full 121-frame export required for intended full-sequence reconstruction.',
          'Sparse source observation in time; no optimization or rendering run.', 'BiRefNet masks and consensus geometry require quality review.',
          'SMPL-X initialization not used. Waymo conditioning retains motion_world_to_canonical_world for later alignment work.']}
    MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'points':len(pos),'frame_checks':checks}),flush=True)


if __name__=='__main__':main()
