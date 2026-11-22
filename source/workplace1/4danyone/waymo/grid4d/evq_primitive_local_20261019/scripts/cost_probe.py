from preflight import *
# Execution-equivalent chunks only; no quality tuning, no optimizer updates.
def main():
 verify();torch.set_num_threads(2);g,d,q,cfg,data,Z=mature('N06','G2');inp=Inputs('N06',data,True);pipe=SimpleNamespace(**cfg['pipeline']);rows=[];refgrads=None;refout=None
 for size in [1024,4096,8192]:
  q.chunk=size;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();q.zero_grad(set_to_none=True);d.grid4d.zero_grad(set_to_none=True);d.deform.zero_grad(set_to_none=True)
  for k in PARAMETERS:getattr(g,k).grad=None
  torch.cuda.synchronize();tick=time.monotonic();state=geometry(g,d,q,data,22,18001,inp,regularize=True);c,a,aa=inp.target(1,22);c2,b,bb=inp.target(7,22);rgb=raster(g,pipe,c,state)['render'];rgb2=raster(g,pipe,c2,state)['render'];alpha=raster(g,pipe,c,state,0,True)['render'];alpha2=raster(g,pipe,c2,state,0,True)['render'];loss=.5*((rgb-a).abs().mean()+(rgb2-b).abs().mean())+.05*((alpha-aa).abs().mean()+(alpha2-bb).abs().mean())+state['reg'];loss.backward();torch.cuda.synchronize();seconds=time.monotonic()-tick;grad={k:v.grad.detach().clone() for k,v in q.named_parameters() if v.grad is not None};out=state['dc_rgb_before_gate'].detach().clone();err=0.;gerr=0.
  if refgrads is None:refgrads=grad;refout=out
  else:
   err=float((out-refout).abs().max());assert err<=1e-5
   for k in grad:
    e=float((grad[k]-refgrads[k]).abs().max());tol=1e-5+1e-3*float(refgrads[k].abs().max());assert e<=tol,(size,k,e,tol);gerr=max(gerr,e)
  rows.append(dict(chunk=size,points=len(g.get_xyz),seconds=seconds,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),forward_max=err,gradient_max=gerr))
  del state,loss,rgb,rgb2,alpha,alpha2,grad,out
 save(NEW/'research/chunk_cost_probe.json',dict(rows=rows,no_optimizer_updates=True,full_21_view_images_cached=True,all32tokens=True,single_timing_not_universal_throughput=True,selected=min(rows,key=lambda r:r['seconds'])['chunk']));print(rows,flush=True)
if __name__=='__main__':main()
