"""Exercise actual second-stage correction on the short-run model, without modifying it."""
import argparse,json,random,time
import numpy as np
import torch
from common import ROOT,WaymoScene,read_image,device,save,stamp
from train_case import configuration
from scene import init_model
from gaussian_renderer import render
import utils.stereo_helper as stereo

@torch.no_grad()
def main():
    cli=argparse.ArgumentParser();cli.add_argument('--run-id',default='original');args_cli=cli.parse_args()
    started=time.monotonic();gpu=device()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    args,data,opt,pipe,config=configuration('N04',2)
    args.model_path=str(ROOT/'tmp'/f'cec_probe_{args_cli.run_id}');data.model_path=args.model_path
    (ROOT/'tmp'/f'cec_probe_{args_cli.run_id}').mkdir(parents=True,exist_ok=False)
    model=init_model(data.model,data);scene=WaymoScene(data,model)
    model.restore(torch.load(ROOT/'smoke/N04/stage1/asset.pt',weights_only=False)[0],opt)
    frame_limit=json.loads((ROOT/'smoke/N04/stage1/progress.json').read_text())['sampling_frame_limit']
    scene.set_sampling_len(frame_limit);stereo.im_reader=read_image
    before=len(model._xyz);selection=[];original=stereo.sample_cameras
    def sample(*a,**kw):
        cams=original(*a,**kw)
        selection.append([{'view':c.colmap_id,'frame':c.timestamp} for c in cams]);return cams
    stereo.sample_cameras=sample
    print(json.dumps({'state':'correcting','points_before':before,'frame_limit':frame_limit}),flush=True)
    result=stereo.correct_gaussians(scene.getTCams(),model,scene,pipe,data,torch.ones(3,device='cuda'),render,0,
        iteration=27385,depth_kernel=opt.cec_depth_kernel,margin_depth=opt.cec_margin_depth,
        error_abs_thres=opt.cec_error_abs_thres,error_rel_thres=opt.cec_error_rel_thres,
        opacity_threshold=opt.cec_opacity_threshold,dynamic_leniency=opt.cec_dynamic_leniency,
        gaussian_blur_size=opt.cec_gaussian_blur_size,nearest_threshold=opt.cec_nearest_threshold)
    assert selection and all(c['view'] not in [0,8,16] for pair in selection for c in pair)
    assert torch.isfinite(model._xyz).all() and torch.isfinite(model._xyz_motion).all()
    cam=scene.getTCams()[0];raw=render(cam,model,pipe,torch.zeros(3,device='cuda'),near=.01,far=100.)
    assert torch.isfinite(raw['render']).all() and float(raw['acc'].max())>0
    import faiss
    record={'state':'passed','run_id':args_cli.run_id,'checked_at':stamp(),'device':gpu,'elapsed_seconds':time.monotonic()-started,
        'points_before':before,'points_after':len(model._xyz),'no_candidates':result is None,
        'faiss_version':faiss.__version__,'faiss_module':faiss.__file__,
        'reported_new_gaussians':None if result is None else int(result[1]),'selected_cameras':selection,
        'scope':'One real CEC call on an immutable short-run checkpoint; not quality evidence or full-stage validation'}
    save(ROOT/'research'/f'cec_probe_{args_cli.run_id}.json',record)
    if args_cli.run_id=='installed':save(ROOT/'research/cec_probe.json',record)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
