"""Compare native acc gradients with identical unit-color RGB and finite differences."""
import math,json
import torch
from common import ROOT,device,save,stamp
from diff_gaussian_rasterization_df import GaussianRasterizationSettings,GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix

def main():
    gpu=device();n=64;fov=math.pi/3
    view=torch.eye(4,device='cuda')
    proj=getProjectionMatrix(.01,10.,fov,fov).transpose(0,1).cuda()
    settings=GaussianRasterizationSettings(image_height=n,image_width=n,tanfovx=math.tan(fov/2),
        tanfovy=math.tan(fov/2),kernel_size=.1,subpixel_offset=torch.zeros(n,n,2,device='cuda'),
        bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=view,projmatrix=proj,
        sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,min_depth=.01,max_depth=10.,debug=False)
    raster=GaussianRasterizer(raster_settings=settings)
    seed={'xyz':torch.tensor([[-.03,.02,2.],[.02,-.01,2.3]],device='cuda'),
          'opacity':torch.tensor([[.2],[.4]],device='cuda'),
          'scale':torch.tensor([[.12,.18,.1],[.16,.13,.1]],device='cuda')}
    # Use interior pixels so perturbations do not cross the rasterizer's hard
    # alpha cutoff. A whole-image finite difference included support changes.
    weights=torch.zeros(n,n,device='cuda')
    weights[28:36,28:36]=torch.linspace(.5,1.5,8,device='cuda')[None,:]*(n*n/64)
    def output(values,kind):
        color,_,_,_,acc,_=raster(means3D=values['xyz'],
            means2D=torch.zeros(2,3,device='cuda'),dir3D=torch.zeros(2,3,device='cuda'),
            colors_precomp=torch.ones(2,3,device='cuda'),opacities=values['opacity'],
            scales=values['scale'],rotations=torch.tensor([[1.,0.,0.,0.]]*2,device='cuda'))
        return ((acc[0] if kind=='native_acc' else color[0])*weights).mean(),color,acc
    gradients={};forward_error=None
    for kind in ['native_acc','unit_rgb']:
        values={key:value.clone().requires_grad_(True) for key,value in seed.items()}
        loss,color,acc=output(values,kind)
        forward_error=float((color[0]-acc[0]).abs().max())
        assert forward_error<1e-6
        grad=torch.autograd.grad(loss,list(values.values()))
        gradients[kind]={k:g.detach() for k,g in zip(values,grad)}
    rows=[]
    for name,index in [('opacity',(0,0)),('opacity',(1,0)),('xyz',(0,0)),('scale',(0,1))]:
        finite=[]
        for epsilon in [.001,.0003]:
            with torch.no_grad():
                plus={k:v.clone() for k,v in seed.items()};minus={k:v.clone() for k,v in seed.items()}
                plus[name][index]+=epsilon;minus[name][index]-=epsilon
                fd=float((output(plus,'native_acc')[0]-output(minus,'native_acc')[0])/(2*epsilon))
            finite.append({'epsilon':epsilon,'derivative':fd})
        expected=finite[-1]['derivative'];unit=float(gradients['unit_rgb'][name][index]);native=float(gradients['native_acc'][name][index])
        rows.append({'parameter':name,'index':index,'native_acc_gradient':native,
            'unit_rgb_gradient':unit,'finite_difference':finite,
            'unit_rgb_relative_error':abs(unit-expected)/max(abs(expected),1e-8),
            'native_acc_relative_error':abs(native-expected)/max(abs(expected),1e-8)})
    save(ROOT/'research/alpha_gradient_probe.json',dict(updated_at=stamp(),device=gpu,
        state='native_acc_backward_mismatch_confirmed' if any(r['native_acc_relative_error']>.2 for r in rows) else 'no_mismatch',
        unit_rgb_backward_check='passed' if max(r['unit_rgb_relative_error'] for r in rows)<.05 else 'failed',
        forward_max_error=forward_error,rows=rows,
        scope='Two overlapping CUDA Gaussians, central8x8 support-stable weighted pixels. Native acc and white-unit-color rendering on black have equal forward alpha; baseline RGB training did not use an alpha loss.',
        recommendation='Do not directly add mask loss to native acc. A separately validated unit-color RGB alpha pass is an explicit differentiable alternative; no change to existing checkpoints or metrics.'))
    assert max(r['unit_rgb_relative_error'] for r in rows)<.05,rows
    assert any(r['native_acc_relative_error']>.2 for r in rows),rows
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
