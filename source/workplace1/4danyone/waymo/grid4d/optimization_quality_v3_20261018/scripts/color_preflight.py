"""Native direction, contribution and coefficient-gradient checks before SH fitting."""
from color_common import *

def fd_check(model,deform,pipe,cam,n=4):
    image=package(model,deform,pipe,cam)['render'];loss=(image-cam.original_image).square().double().mean()
    gradient=torch.autograd.grad(loss,model._features_dc)[0];indices=gradient.flatten().abs().topk(n).indices.tolist();rows=[]
    for index in indices:
        flat=model._features_dc.view(-1);original=float(flat[index]);eps=.01
        with torch.no_grad():
            flat[index]=original+eps;plus=float((package(model,deform,pipe,cam)['render']-cam.original_image).double().square().mean())
            flat[index]=original-eps;minus=float((package(model,deform,pipe,cam)['render']-cam.original_image).double().square().mean());flat[index]=original
        fd=(plus-minus)/(2*eps);analytic=float(gradient.flatten()[index]);error=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12)
        assert error<.01,(fd,analytic,error);rows.append(dict(index=index,analytic=analytic,finite_difference=fd,relative_error=error,epsilon=eps))
    return rows

def main():
    begin=time.monotonic();device=verify();sys.addaudithook(protect_inputs);source,model,deform,pipe,config,data,frozen=setup();obs=pool(data)
    references=np.load(source/'snapshots/53000_reference.npz');now=asset_reference(model,deform,pipe,data)
    replay=max(float(np.abs(now[k]-references[k]).max()) for k in references.files);assert replay<1e-5
    rows=[]
    for index in [0,7]:
        cam,path=obs[index];native=package(model,deform,pipe,cam);basis=basis_for(model,cam,native['world_means']);colors=torch.einsum('nk,nkc->nc',basis,model.get_features)+.5
        custom=package(model,deform,pipe,cam,colors.clamp_min(0));difference=float((custom['render']-native['render']).abs().max());assert difference<1e-5
        constant=torch.full_like(model.get_xyz,.25,requires_grad=True);const=package(model,deform,pipe,cam,constant)['render']
        with torch.no_grad():
            white=package(model,deform,pipe,cam,torch.ones_like(model.get_xyz))['render'];alpha=2*(white[0]-.5)
        composition=float((const-(.25*alpha+.5*(1-alpha))[None]).abs().max());assert composition<1e-5
        derivative=torch.autograd.grad(const[0].sum(),constant)[0][:,0];conservation=float(abs(derivative.sum()-alpha.sum())/alpha.sum());assert conservation<1e-5
        # Actual native SH and explicitly precomputed SH must give the same SH derivative.
        grad_native=torch.autograd.grad((native['render']-cam.original_image).square().mean(),[model._features_dc,model._features_rest])
        grad_custom=torch.autograd.grad((custom['render']-cam.original_image).square().mean(),[model._features_dc,model._features_rest])
        grad_error=max(float((a-b).abs().max()) for a,b in zip(grad_native,grad_custom));assert grad_error<1e-6
        rows.append(dict(view=cam.source_view,frame=cam.source_frame,native_direction='deformed_world_mean-minus-camera_center',color_max_abs_error=difference,gradient_max_abs_error=grad_error,constant_composition_error=composition,contribution_conservation_relative_error=conservation,negative_preclamp_channels=int((colors<0).sum()),total_channels=colors.numel(),real_coefficient_fd=fd_check(model,deform,pipe,cam)))
    # Two overlapping splats with deterministic, visible world positions, no deformation.
    cam=obs[0][0];weights,basis=contribution(model,deform,pipe,cam);ids=weights.topk(2).indices;pkg=package(model,deform,pipe,cam)
    tiny=GaussianModel(3)
    for name in PARAMETERS:setattr(tiny,name,torch.nn.Parameter(getattr(model,name)[ids].detach().clone(),requires_grad=name in ['_features_dc','_features_rest']))
    tiny.active_sh_degree=3
    with torch.no_grad():
        tiny._xyz.copy_(pkg['world_means'][ids]);tiny._xyz[1].copy_(tiny._xyz[0]+torch.tensor([.002,0.,0.],device='cuda'));tiny._scaling.fill_(math.log(.02));tiny._opacity.fill_(math.log(.7/.3));tiny._features_dc.fill_(.1);tiny._features_rest.zero_()
    class Static:
        def step(self,*a):return dict(d_xyz=0.,d_rotation=0.,d_scaling=0.)
    tiny_fd=fd_check(tiny,Static(),pipe,cam,n=4)
    verify_frozen(model,deform,frozen)
    save(NEW/'color_preflight.json',dict(state='passed',device=device,zero_update_replay_max_error=replay,native_branch='CUDA SH, clamp_min(SH+.5,0)',observations=rows,tiny_overlapping_gaussians_fd=tiny_fd,normalization='RGB MSE normalized by3*H*W; full pool divided by168; preconditioner2/(3*H*W*168)',contribution='first-order V=sum_pixels(T*alpha), not squared contribution or Hessian',source_sha256=sha(source/'point_cloud/iteration_53000/point_cloud.ply'),seconds=time.monotonic()-begin))
if __name__=='__main__':
    import math
    main()
