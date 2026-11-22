"""Exercise the actual third-party parser, RGBA dataset, and NPZ time conversion."""
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BACKEND=ROOT/'bridge/backend'
LABEL=os.environ.get('FDANYONE_BRIDGE_LABEL','official_sample')
assert LABEL in ('official_sample','waymo_sample')
SOURCE=Path(os.environ.get('FDANYONE_BRIDGE_SOURCE',str(ROOT.parent/'outputs/turbo')))
sys.path.insert(0,str(BACKEND));sys.path.insert(0,str(BACKEND/'src'))
from datasets.FreeTime_dataset import FreeTimeParser,FreeTimeDataset
from init_common import read_init_arrays,apply_scene_transform


def main():
    dest=ROOT/'bridge'/LABEL
    cameras=json.loads((SOURCE/'cameras.json').read_text())['cameras']
    raw=FreeTimeParser(str(dest),normalize=False,start_frame=0,end_frame=3)
    normalized=FreeTimeParser(str(dest),normalize=True,start_frame=0,end_frame=3)
    assert raw.camera_names==[f'cam{i:02d}' for i in range(24)]
    errors=[]
    for i,camera in enumerate(cameras):
        assert np.allclose(raw.camtoworlds[i],camera['camera_to_world'],atol=1e-6)
        K=np.array(camera['K']);K[:2]*=.25
        assert np.allclose(raw.Ks_dict[i+1],K,atol=1e-6)
        assert abs(np.linalg.det(raw.camtoworlds[i,:3,:3])-1)<1e-6
        # A true round trip through a camera ray catches axis flips and translation errors.
        uv=np.array([[20,40,1],[88,160,1],[150,290,1]],float)
        xyz=(uv@np.linalg.inv(K).T)*3
        world=xyz@raw.camtoworlds[i,:3,:3].T+raw.camtoworlds[i,:3,3]
        transformed,_=apply_scene_transform(world,np.zeros_like(world),normalized.transform)
        inverse=np.linalg.inv(normalized.camtoworlds[i])
        cam=transformed@inverse[:3,:3].T+inverse[:3,3]
        projected=cam@K.T;projected=projected[:,:2]/projected[:,2:]
        errors.append(float(np.abs(projected-uv[:,:2]).max()))
    assert max(errors)<.001,errors
    train=FreeTimeDataset(normalized,split='train',test_set=[0,8,16],alpha_mode='transparent')
    test=FreeTimeDataset(normalized,split='test',test_set=[0,8,16],alpha_mode='transparent')
    assert len(train)==63 and len(test)==9
    for dataset in (train,test):
        for idx in range(len(dataset)):
            sample=dataset[idx]
            assert tuple(sample['image'].shape)==(320,176,3)
            assert tuple(sample['alpha'].shape)==(320,176,1)
            assert 0<float(sample['alpha'].mean())<1
            assert sample['time']==sample['frame_offset']/2
    init=read_init_arrays(str(dest/'init.npz'),0,3)
    assert set(init['times'].tolist())=={0,.5,1}
    pos,vel=apply_scene_transform(init['positions'],init['velocities'],normalized.transform)
    assert np.isfinite(pos).all() and np.max(np.abs(vel))==0
    # Nonzero controlled fixture verifies the backend's meters/frame -> normalized-time rule.
    fixture=ROOT/'bridge'/f'{LABEL}_time_fixture.npz'
    np.savez(fixture,positions=np.zeros((3,3),np.float32),colors=np.ones((3,3),np.float32),velocities=np.ones((3,3),np.float32)*.1,
             times=np.array([0,.5,1],np.float32),durations=np.ones(3,np.float32)*.25,frame_start=0,frame_end=3,time_denominator=2)
    trial=read_init_arrays(str(fixture),0,3)
    assert np.allclose(trial['velocities'],.2) and np.allclose(trial['durations'],.25)
    results={'status':'passed_actual_backend_loader','backend_commit':'1f160e290b00d88e097739ecc26d93500d6c55e8',
             'python':sys.version,'train_samples':len(train),'test_samples':len(test),'rgba_samples_checked':72,
             'camera_count':24,'normalization_reprojection_max_error_pixels':max(errors),
             'backend_normalization_transform':normalized.transform.tolist(),'initialization_points':len(pos),
             'nonzero_velocity_unit_test':'0.1 meters/export-frame becomes 0.2 meters/normalized-time for 3-frame sequence',
             'dependencies':{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','pycolmap','imageio']},
             'scope':'CPU loader and geometry/time checks only. CUDA trainer extensions, optimization and render remain untested.'}
    validation=ROOT/'bridge/validation.json' if LABEL=='official_sample' else ROOT/'bridge/waymo_validation.json'
    validation.write_text(json.dumps(results,indent=2)+'\n')
    manifest_path=ROOT/'bridge_manifest.json' if LABEL=='official_sample' else ROOT/'bridge/waymo_manifest.json'
    manifest=json.loads(manifest_path.read_text());manifest['status']='export_and_actual_loader_validated';manifest['validation']=str(validation.relative_to(ROOT));manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
