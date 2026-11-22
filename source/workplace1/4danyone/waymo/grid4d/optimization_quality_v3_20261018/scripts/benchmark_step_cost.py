"""Read-only paired render/backward cost probe on one idle GPU; no optimizer steps."""
from quality_common import *
def main():
    device=verify();sys.addaudithook(protect_inputs);records=[]
    for label in ['N01','N07']:
        source=NEW/'runs'/f'{label}_A';model,deform,pipe,config,data,state=restore(source);full,_=manifest(label);observations=[]
        for view in [1,9,17]:
            for frame in [0,(data['frame_count']-1)//2,data['frame_count']-1]:
                path=Path(full['source_data'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
                with Image.open(path) as im:rgba=torch.tensor(np.asarray(im).copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
                observations.append((camera(data,view,frame,rgba[:3]*rgba[3:]+.5*(1-rgba[3:])),rgba[3]))
        opt=SimpleNamespace(**config['optimization']);rows=[]
        def one(cam,target,with_alpha):
            model.optimizer.zero_grad(set_to_none=True);deform.optimizer.zero_grad(set_to_none=True);set_rng(state['rng']);torch.cuda.synchronize();start=time.monotonic()
            warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(len(model.get_xyz),-1));image=render(cam,model,pipe,torch.full((3,),.5,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])['render']
            loss=.8*(image-cam.original_image).abs().mean()+.2*(1-ssim(image,cam.original_image))+warp['reg_spatial'].mean()*opt.lambda_spatial_tv+warp['reg_temporal'].mean()*opt.lambda_temporal_tv
            if with_alpha:
                alpha=render(cam,model,pipe,torch.zeros(3,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0];loss=loss+.1*old_alpha(alpha,target)
            loss.backward();torch.cuda.synchronize();return time.monotonic()-start
        for mode in [False,True]:one(*observations[0],mode)
        for repeat in range(3):
            for cam,target in observations:
                for mode in ([False,True] if repeat%2==0 else [True,False]):rows.append(dict(repeat=repeat,view=cam.source_view,frame=cam.source_frame,alpha=mode,seconds=one(cam,target,mode)))
        for name in PARAMETERS:equal(getattr(model,name),state['gaussians'][name])
        equal(deform.grid4d.state_dict(),state['hash']);equal(deform.deform.state_dict(),state['deform']);means={str(mode):float(np.mean([r['seconds'] for r in rows if r['alpha']==mode])) for mode in [False,True]};estimate=(15000*means['False']+18000*means['True'])/(33000*means['False']+20000*means['True'])
        records.append(dict(label=label,device=device,source=str(source),points=len(model.get_xyz),rows=rows,means=means,alpha_to_rgb_ratio=means['True']/means['False'],representative_recipe_cost_ratio=estimate,optimizer_updates=0,parameters_unchanged=True))
        del model,deform,state,observations;torch.cuda.empty_cache()
    save(NEW/'research/step_cost_probe.json',dict(state='passed',records=records,scope='Paired same final geometry and nine fixed training observations, synchronized wall time of render/loss/backward only. Includes identical feature regularization; excludes optimizer, densification, setup, IO and evaluation. Three repeats, AB/BA order.',interpretation='Representative computation estimate, not a measured full-pipeline speedup and not a quality experiment. Full training wall remains confounded by shared load and migration.'))
if __name__=='__main__':main()
