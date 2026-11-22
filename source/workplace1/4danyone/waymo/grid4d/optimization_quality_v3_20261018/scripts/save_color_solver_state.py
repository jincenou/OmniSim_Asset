"""Reconstitute frozen-geometry solver state for future recovery; no color updates."""
from color_common import *
def main():
    begin=time.monotonic();device=verify();sys.addaudithook(protect_inputs);source,model,deform,pipe,config,data,frozen=setup();observations=pool(data);k=16
    blocks=torch.zeros(len(model.get_xyz),k,k,device='cuda',dtype=torch.float64)
    for cam,path in observations:
        v,y=contribution(model,deform,pipe,cam);blocks.add_(v.double()[:,None,None]*y.double()[:,:,None]*y.double()[:,None,:])
    blocks.mul_(2/(3*data['height']*data['width']*168));trace=blocks.diagonal(dim1=-2,dim2=-1).sum(-1);damping=trace/k*1e-4;blocks.diagonal(dim1=-2,dim2=-1).add_(damping[:,None]);chol,info=torch.linalg.cholesky_ex(blocks);valid=(trace>0)&(info==0)
    old=json.loads((NEW/'runs/N11_C2/block_statistics.json').read_text());assert int(valid.sum())==old['valid_blocks'];verify_frozen(model,deform,frozen)
    target=NEW/'runs/N11_C2/recoveries/solver_state_reconstituted.pt';assert not target.exists();atomic_torch(target,dict(cholesky=chol,valid=valid,step=.5,trace_relative_damping=1e-4,normalization=2/(3*data['height']*data['width']*168),basis='native CUDA deformed-world SH degree3',geometry_source=str(source),pool=json.loads((NEW/'runs/N11_C2/color_pool.json').read_text()),provenance='Reconstituted from frozen geometry and original algorithm; original temporary GPU matrix was not serialized. No new optimizer updates.'))
    save(NEW/'runs/N11_C2/recoveries/solver_state.json',dict(state='available',path=target.name,sha256=sha(target),seconds=time.monotonic()-begin,device=device,optimizer_updates=0,valid_blocks=int(valid.sum()),normalization=2/(3*data['height']*data['width']*168),limits='Not claimed byte-identical to original unrecorded GPU reduction; source geometry, pool, weighting, damping and solver unchanged. Rendering does not require this state.'))
if __name__=='__main__':main()
