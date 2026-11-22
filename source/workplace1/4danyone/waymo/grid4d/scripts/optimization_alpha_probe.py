"""Real-rasterizer alpha gradient acceptance before supervised optimization."""
import torch
from grid_common import ROOT, load_asset, manifest, camera, render, save, verify


def main():
    device=verify();model,deform,pipe,_=load_asset(ROOT/'runs/N11_33k_v1',33000)
    data,_=manifest('N11');cam=camera(data,1,0)
    warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(len(model.get_xyz),-1))
    def output():
        return render(cam,model,pipe,torch.zeros(3,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0]
    alpha=output();value=alpha.mean()
    grad=torch.autograd.grad(value,model._opacity)[0];analytic=float(grad.sum())
    original=model._opacity.detach().clone();epsilon=.001
    with torch.no_grad():
        model._opacity.copy_(original+epsilon);plus=float(output().mean())
        model._opacity.copy_(original-epsilon);minus=float(output().mean())
        model._opacity.copy_(original)
    numeric=(plus-minus)/(2*epsilon);relative=abs(analytic-numeric)/max(abs(numeric),1e-12)
    assert analytic>0 and numeric>0 and relative<.01,(analytic,numeric,relative)
    save(ROOT/'optimization/alpha_gradient_probe.json',dict(state='passed',device=device,scope='Real N11 terminal geometry,training view1/frame0; directional derivative of mean true alpha wrt all opacity logits',analytic=analytic,finite_difference=numeric,relative_error=relative,tolerance=.01,epsilon=epsilon,original_assets_modified=False))
    print('alpha gradient probe passed',analytic,numeric,relative,flush=True)


if __name__=='__main__':main()
