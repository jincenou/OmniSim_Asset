"""Validate real CUDA forward/backward, camera projection, GPU KNN and native replay."""
import json,os
import numpy as np
import torch
import faiss
from common import ROOT,REPO,manifest,camera,device,save,sha,stamp
from train_case import configuration
from scene import init_model
from utils.graphics_utils import BasicPointCloud
from gaussian_renderer import render

def main():
    print('imports_ok; checking CUDA device',flush=True)
    gpu=device();torch.set_num_threads(2)
    assert os.environ['CONDA_PREFIX']!='/mnt/ojc/miniconda3/envs/4dgaussians' if 'CONDA_PREFIX' in os.environ else True
    resource=faiss.StandardGpuResources();index=faiss.GpuIndexFlatL2(resource,3)
    values=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32);index.add(values)
    distances,ids=index.search(values,1);assert np.array_equal(ids[:,0],np.arange(3)) and distances.max()==0
    print('GPU FAISS checked; hashing fixed input images',flush=True)
    checked={};max_projection=0.
    for label in ['N04','N08','N11']:
        data,path=manifest(label)
        for item in data['input_hashes']:
            p=__import__('pathlib').Path(data['source_data'])/'images'/f"cam{item['view']:02d}"/f"{item['frame']:06d}.png"
            assert sha(p)==item['sha256'],str(p)
        checked[label]={'manifest_sha256':sha(path),'image_hashes_checked':len(data['input_hashes'])}
        print(json.dumps({'label':label,'input_hashes_checked':len(data['input_hashes'])}),flush=True)
        for stage in [1,2]: save(ROOT/'configs'/f'{label}_stage{stage}.json',configuration(label,stage)[-1])
        with np.load(path.parent/data['canonical_npz']) as z: xyz=z['points'][:128].astype(np.float64)
        homogeneous=np.c_[xyz,np.ones(len(xyz))]
        for v in range(24):
            cam=camera(data,v,0);record=data['cameras'][v]
            K=np.asarray(record['K']);w=np.linalg.inv(np.asarray(record['camtoworld']))
            cv=homogeneous@w.T;uv=cv[:,:3]@K.T;uv=uv[:,:2]/uv[:,2:]
            ndc=homogeneous@cam.full_proj_transform.cpu().numpy();ndc=ndc[:,:2]/ndc[:,3:]
            pixels=((ndc+1)*np.array([352,640])-1)/2
            error=float(np.max(np.abs(pixels-uv)));assert error<.002,(label,v,error)
            max_projection=max(max_projection,error)
    args,dataset,opt,pipe,config=configuration('N04',1,True)
    data,path=manifest('N04')
    with np.load(path.parent/data['canonical_npz']) as z:
        cloud=BasicPointCloud(z['points'],z['colors'],np.zeros_like(z['points']),np.zeros((len(z['points']),1)))
    model=init_model(dataset.model,dataset);model.create_from_pcd(cloud,data['scene_radius']);model.training_setup(opt)
    cam=camera(data,1,0);bg=torch.zeros(3,device='cuda')
    before=model._xyz.detach().clone();pack=render(cam,model,pipe,bg,near=.01,far=100.)
    assert pack['render'].shape==(3,640,352) and torch.isfinite(pack['render']).all()
    assert float(pack['acc'].max())>0 and pack['acc'].shape==(1,640,352)
    loss=pack['render'].square().mean();loss.backward()
    assert model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all() and float(model._xyz.grad.abs().max())>0
    model.optimizer.step();assert float((model._xyz-before).abs().max())>0
    model.optimizer.zero_grad(set_to_none=True)
    near=model.get_nearest_gaussian(model._xyz[:3].detach(),0.,opacity_thres=.01)
    assert (near>=0).all()
    torch.save((model.capture(),1),ROOT/'research/preflight_checkpoint.pt')
    clone=init_model(dataset.model,dataset);clone.restore(torch.load(ROOT/'research/preflight_checkpoint.pt',weights_only=False)[0],opt)
    with torch.no_grad():
        a=render(cam,model,pipe,bg,near=.01,far=100.);b=render(cam,clone,pipe,bg,near=.01,far=100.)
    replay=max(float((a[k]-b[k]).abs().max()) for k in ['render','acc'])
    assert replay<1e-6,replay
    result={'state':'passed','device':gpu,'checked_at':stamp(),'inputs':checked,'projection_max_pixels':max_projection,
        'native_rasterizer_cuda_backward':True,'native_gpu_nearest_neighbor':True,'capture_restore_max_error':replay,
        'scope':'runtime_and_input_interface_only_not_training_or_quality','environment':'/mnt/ojc/miniconda3/envs/cem4dgs'}
    save(ROOT/'research/preflight.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
