"""Local EVQ helpers; original Grid4D/runtime and frozen input remain read-only."""
import sys,json,hashlib,os,random,time
from pathlib import Path
from types import SimpleNamespace
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;BASE=ROOT.parent/'random8/r20261015_113332'
sys.path.insert(0,str(ROOT/'scripts'))
from grid_common import *
from utils.general_utils import quaternion_to_matrix
from utils.sh_utils import C0
NEW=Path(__file__).resolve().parents[1]

def native_world(xyz,warp):
    """Exactly the shared native quaternion+translation transform, not additive d_xyz."""
    if not torch.is_tensor(warp):return xyz
    assert warp.shape==(len(xyz),7)
    return (quaternion_to_matrix(warp[:,:4])@xyz[...,None]+warp[:,4:,None]).squeeze(-1)

def base_features(deform,xyz,t,regularize=False):
    previous=deform.reg_spatial_able,deform.reg_temporal_able
    deform.reg_spatial_able=previous[0] and regularize;deform.reg_temporal_able=previous[1] and regularize
    result=[];hook=deform.grid4d.register_forward_hook(lambda m,a,o:result.append(o))
    try:warp=deform.step(xyz,t.expand(len(xyz),1));assert len(result)==1;features=torch.cat(result[0],dim=-1)
    finally:hook.remove();deform.reg_spatial_able,deform.reg_temporal_able=previous
    return warp,features

def guard_training(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(BASE):assert not (args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)), 'Original input writes forbidden'
        assert not (p.is_relative_to(BASE) and p.suffix.lower() in ['.png','.jpg','.npz','.mp4'] and any(f'/cam{v:02d}/' in str(p) for v in [0,8,16])),'H content forbidden during training'

def atomic_torch(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.writing');torch.save(payload,tmp);tmp.replace(path)

def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b
