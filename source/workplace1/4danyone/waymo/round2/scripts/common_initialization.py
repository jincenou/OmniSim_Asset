"""CPU common-six-view initialization with no heldout geometry/color leakage."""
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT=Path(__file__).resolve().parents[1];W=ROOT.parent;R=W/'reconstruction'
H=[0,8,16];V6=[1,5,9,13,17,21];V12=list(range(1,24,2));V21=[i for i in range(24) if i not in H]
assert set(V6)<set(V12)<set(V21) and not set(V21)&set(H)
sys.path[:0]=[str(W/'bridge/backend'),str(W/'bridge/backend/src')]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def project(points,camera):
    w=np.linalg.inv(np.asarray(camera['camera_to_world']));p=points@w[:3,:3].T+w[:3,3]
    k=np.asarray(camera['K'],dtype=np.float64).copy();k[:2]*=.5;q=p@k.T
    uv=np.rint(q[:,:2]/q[:,2:]).astype(int)
    valid=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<352)&(uv[:,1]>=0)&(uv[:,1]<640)
    return uv,valid

def prepare(label):
    record=R/'research'/('full-export.json' if label=='phone_full' else f'{label}-export.json')
    origin=json.loads(record.read_text());source=Path(origin['source']);data=R/'data'/label
    destination=ROOT/'controls'/label/'common6';destination.mkdir(parents=True,exist_ok=False)
    cams=json.loads((source/'cameras.json').read_text())['cameras']
    assert len(cams)==24 and [int(c['camera_id']) for c in cams]==list(range(24))
    shape=(64,96,64);axes=np.meshgrid(np.linspace(-1.2,1.2,64),np.linspace(-.1,2.1,96),np.linspace(-1.2,1.2,64),indexing='ij')
    grid=np.stack(axes,-1).reshape(-1,3);projections={i:project(grid,cams[i]) for i in V6}
    positions=[];colors=[];times=[];evidence=[];inputs=[]
    for frame in range(0,121,15):
        images={}
        for i in V6:
            path=data/'images'/f'cam{i:02d}'/f'{frame:06d}.png'
            with Image.open(path) as im:images[i]=np.array(im)
            assert images[i].shape==(640,352,4)
            inputs.append({'camera_id':i,'frame':frame,'path':str(path),'sha256':sha(path)})
        hits=np.zeros(len(grid),np.int16)
        for i in V6:
            uv,valid=projections[i];inside=np.zeros(len(grid),bool);inside[valid]=images[i][uv[valid,1],uv[valid,0],3]>=128;hits+=inside
        occupied=(hits>=6).reshape(shape)
        idx=np.flatnonzero((occupied&~binary_erosion(occupied)).ravel())
        if len(idx)<50:raise RuntimeError(f'{label} common-six hull insufficient at {frame}: {len(idx)}')
        pts=grid[idx];rgb=np.zeros((len(pts),3));counts=np.zeros(len(pts))
        for i in V6:
            uv,valid=project(pts,cams[i]);valid[valid]&=images[i][uv[valid,1],uv[valid,0],3]>=128
            rgb[valid]+=images[i][uv[valid,1],uv[valid,0],:3];counts[valid]+=1
        assert counts.min()>0
        positions.append(pts.astype(np.float32));colors.append((rgb/counts[:,None]).astype(np.float32));times.append(np.full(len(pts),frame/120,np.float32));evidence.append({'frame':frame,'points':len(pts)})
    pos=np.concatenate(positions);rgb=np.concatenate(colors)
    init=destination/'init.npz'
    np.savez_compressed(init,positions=pos,colors=rgb,velocities=np.zeros_like(pos),times=np.concatenate(times),durations=np.full(len(pos),.125,np.float32),frame_start=0,frame_end=121,time_denominator=120,keyframe_step=15,mode='common_six_training_views_only_nine_hull_times')
    sparse=destination/'sparse/0';sparse.mkdir(parents=True)
    shutil.copyfile(data/'sparse/0/cameras.txt',sparse/'cameras.txt');shutil.copyfile(data/'sparse/0/images.txt',sparse/'images.txt')
    # Parser PCA depends on points3D; never reuse the original 21-view hull here.
    lines=[f'{i+1} {p[0]:.8g} {p[1]:.8g} {p[2]:.8g} {int(c[0])} {int(c[1])} {int(c[2])} 0' for i,(p,c) in enumerate(zip(positions[0],colors[0],strict=True))]
    (sparse/'points3D.txt').write_text('\n'.join(lines)+'\n')
    (destination/'images').symlink_to(data/'images',target_is_directory=True)
    from datasets.FreeTime_dataset import FreeTimeParser,FreeTimeDataset
    parser=FreeTimeParser(str(destination),normalize=True,start_frame=0,end_frame=121)
    assert parser.camera_names==[f'cam{i:02d}' for i in range(24)]
    assert parser.camera_ids==list(range(1,25))
    cases=[]
    for train in [V6,V12,V21]:
        remove=sorted(set(range(24))-set(train)-set(H))
        dataset=FreeTimeDataset(parser,split='train',test_set=H,remove_set=remove,alpha_mode='transparent')
        validation=FreeTimeDataset(parser,split='val',test_set=H,alpha_mode='transparent')
        assert dataset.camera_indices==train and validation.camera_indices==H
        assert len(dataset)==len(train)*121 and len(validation)==3*121
        assert set(dataset.pairs).isdisjoint(validation.pairs)
        samples=[dataset[0],dataset[len(dataset)-1],validation[0],validation[len(validation)-1]]
        assert all(tuple(x['image'].shape)==(640,352,3) and tuple(x['alpha'].shape)==(640,352,1) for x in samples)
        case={'train_views':train,'holdout_views':H,'remove_set':remove,'train_samples':len(dataset),'validation_samples':len(validation),'shared_init':str(init),'shared_init_sha256':sha(init),'shared_data_dir':str(destination),'frames':121,'dimensions':[352,640],'optimization_steps':5000,'seed':42,'state':'actual_loader_checked_not_trained'}
        save(destination.parent/f'views{len(train)}.json',case);cases.append(case)
    report={'state':'common_initialization_and_loader_verified_not_trained','label':label,'initialization_views':V6,'holdout_views':H,'points':len(pos),'keyframe_points':evidence,'input_rgba_provenance':inputs,'init_sha256':sha(init),'points3D_sha256':sha(sparse/'points3D.txt'),'normalization_transform':parser.transform.tolist(),'normalization_geometry':'First-frame surface derived only from common6; all24 camera poses retained identically across cases','cases':cases,'gpu_used':False}
    save(destination.parent/'validation.json',report)
    print(json.dumps({'label':label,'state':report['state'],'points':len(pos)}),flush=True)

async def main():
    # Bounded asynchronous subprocesses keep parser imports and artifacts isolated.
    sem=asyncio.Semaphore(1)
    async def one(label):
        async with sem:
            p=await asyncio.create_subprocess_exec(sys.executable,__file__,label)
            if await p.wait():raise RuntimeError('Common initialization failed: '+label)
    await asyncio.gather(*map(one,['phone_full','pilot_043_twentyfour']))

if __name__=='__main__':
    if len(sys.argv)==2:prepare(sys.argv[1])
    else:asyncio.run(main())
