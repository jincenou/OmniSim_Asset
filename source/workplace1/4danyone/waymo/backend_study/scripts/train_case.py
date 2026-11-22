"""Run the author two-stage optimizer with explicit Waymo IO and accounting."""
from argparse import ArgumentParser
from types import SimpleNamespace
import gc,json,os,random,time,subprocess,sys,fcntl
import numpy as np
import torch
from common import ROOT,REPO,LABELS,manifest,WaymoScene,read_image,camera,device,save,sha,stamp
import scene
scene.im_reader=read_image
import train
import utils.stereo_helper as stereo
stereo.im_reader=read_image
from arguments import ModelParams,PipelineParams,OptimizationParams

def configuration(label,stage,smoke=False):
    data,path=manifest(label)
    parser=ArgumentParser();mp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args([])
    official=REPO/'configs'/f'techni_s{stage}'/'Painter.json'
    config=json.loads(official.read_text())
    for k,v in config.items(): setattr(args,k,v)
    args.source_path=str(path.parent);args.label=label;args.loader='waymo_adapter'
    args.duration=data['frame_count'];args.start_timestamp=0;args.end_timestamp=data['frame_count']
    args.start_duration=10 if stage==1 else data['frame_count']
    args.resolution=1;args.white_background=True;args.random_background=False
    args.lazy_loader=True;args.near=.01;args.far=100.
    args.iterations=1200 if smoke else (27000 if stage==1 else 35000)
    args.model_path=str(ROOT/('smoke' if smoke else 'runs')/label/f'stage{stage}')
    args.save_iterations=[];args.checkpoint_iterations=[];args.test_iterations=[]
    args.quiet=False;args.debug_from=-1
    dataset=mp.extract(args);dataset.label=label
    record={'label':label,'stage':stage,'kind':'smoke_not_quality' if smoke else 'frozen_quality_pilot',
        'seed':42,'dataset':vars(dataset),'optimization':vars(op.extract(args)),'pipeline':vars(pp.extract(args)),
        'official_config':str(official),'official_config_sha256':sha(official),
        'source_commit':'7b5dfe3c05456f95c56b3318e406ed6d0f3c05bd',
        'manifest':str(path),'manifest_sha256':sha(path),'training_background':1.,'evaluation_background':.5,
        'official_config_fields_not_in_parameter_groups':sorted(set(config)-set(vars(dataset))-set(vars(op.extract(args)))-set(vars(pp.extract(args)))),
        'time_mapping':'CEM timestamp=f; real_seconds=f/25; baseline normalized_time=f/(F-1)',
        'recipe_note':'Author Painter stages27k/35k. Same21 views, full time, t0 points, fixed white compositing as HUST; no new loss.'}
    return args,dataset,op.extract(args),pp.extract(args),record

def run_stage(label,stage,smoke=False):
    args,dataset,opt,pipe,config=configuration(label,stage,smoke)
    out=ROOT/('smoke' if smoke else 'runs')/label/f'stage{stage}';out.mkdir(parents=True,exist_ok=False)
    checked_device=device();save(out/'config.json',config)
    checkpoint=None
    if stage==1:
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
    else:
        prior=out.parent/'stage1';assert (prior/'complete.json').exists()
        projection=json.loads((ROOT/'research/cec_projection_after.json').read_text())
        assert projection['state']=='passed' and projection['source_sha256']==sha(REPO/'utils/camera_utils.py')
        import faiss
        cec=json.loads((ROOT/'research/cec_probe.json').read_text())
        assert cec['state']=='passed' and cec['faiss_version']==faiss.__version__ and cec['faiss_module']==faiss.__file__,'Full CEC probe must pass using the installed GPU runtime before stage2'
        checkpoint=str(prior/'chkpnt27000.pth')
        rng=torch.load(prior/'rng.pt',weights_only=False)
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
    active={};trace=[];corrections=[];updates=0;started=time.monotonic()
    original_correct=train.correct_gaussians
    def make_scene(*a,**kw):
        result=WaymoScene(*a,**kw);active['scene']=result;return result
    def correct(*a,**kw):
        cams=a[0];assert all(c.colmap_id not in [0,8,16] for c in cams)
        begin=time.monotonic();before=len(a[1]._xyz)
        result=original_correct(*a,**kw)
        # Author helper returns None when no candidate exists; training unconditionally unpacks it.
        record={'iteration':kw['iteration'],'elapsed_seconds':time.monotonic()-begin,
                'points_before':before,'points_after':len(a[1]._xyz),'no_candidates':result is None,
                'candidate_views':sorted(set(c.colmap_id for c in cams))}
        corrections.append(record);save(out/'correction_events.json',corrections)
        return ([],0,[]) if result is None else result
    def after(iteration,cam,model,loss,updated):
        nonlocal updates
        assert np.isfinite(loss) and cam.colmap_id not in [0,8,16]
        trace.append([iteration,cam.colmap_id,int(cam.timestamp)]);updates+=int(updated)
        if iteration==1 or iteration%200==0 or iteration==args.iterations:
            assert torch.isfinite(model._xyz).all() and torch.isfinite(model._scaling).all()
            progress={'state':'training','label':label,'stage':stage,'iteration':iteration,'updates_this_stage':updates,
                'points':len(model._xyz),'motion_groups':len(model._xyz_motion),
                'sampling_frame_limit':active['scene'].sample_len,
                'dynamic_points':int((model._parent[:len(model._xyz)]!=-1).sum()),
                'loss':loss,'elapsed_seconds':time.monotonic()-started,'device':checked_device,'updated_at':stamp()}
            save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    train.Scene=make_scene;train.correct_gaussians=correct;train._study_after_iteration=after
    train.args=args;train.network_gui.listener.close();train.network_gui.try_connect=lambda:None
    # Same final heldout evaluation is performed in a separate process with the shared metric protocol.
    train.training_report=lambda *a,**kw:None
    train.prepare_output_and_logger=lambda args:None
    try:
        train.training(dataset,opt,pipe,[],[],[],checkpoint,-1,args)
    finally:
        train.correct_gaussians=original_correct
    model=active['scene'].gaussians
    expected=(args.iterations-1) if stage==1 else 7999
    assert updates==expected,(updates,expected)
    np.save(out/'sampling.npy',np.asarray(trace,np.int32))
    torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state_all()},out/'rng.pt')
    # Explicit final full checkpoint is the replay authority, even if native PLY omits attributes.
    torch.save((model.capture(),args.iterations),out/'asset.pt')
    data,_=manifest(label);reference={}
    last=min(data['frame_count']-1,int(active['scene'].sample_len))
    if not smoke:assert last==data['frame_count']-1,'Formal training did not reach full temporal window'
    with torch.no_grad():
        for frame in [0,last//2,last]:
            for view in [0,8,16]:
                raw=train.render(camera(data,view,frame),model,pipe,torch.zeros(3,device='cuda'),near=.01,far=100.)
                reference[f'rgb_{view}_{frame}']=raw['render'].cpu().numpy()
                reference[f'alpha_{view}_{frame}']=raw['acc'].cpu().numpy()
    np.savez_compressed(out/'replay_reference.npz',**reference)
    result={'state':'trained_replay_evaluation_pending','label':label,'stage':stage,'kind':config['kind'],
        'updates_this_stage':updates,'nominal_iteration':args.iterations,'training_samples':len(trace),
        'elapsed_seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'points':len(model._xyz),'motion_groups':len(model._xyz_motion),
        'dynamic_points':int((model._parent[:len(model._xyz)]!=-1).sum()),'correction_calls':len(corrections),
        'device':checked_device,'asset_sha256':sha(out/'asset.pt'),'config_sha256':sha(out/'config.json'),
        'ended_at':stamp(),'production_approved':False}
    save(out/'complete.json',result);print(json.dumps(result),flush=True)
    del active,model;gc.collect();torch.cuda.empty_cache()

def main():
    cli=ArgumentParser();cli.add_argument('--label',choices=LABELS,required=True)
    cli.add_argument('--stage',type=int,choices=[1,2],required=True);cli.add_argument('--smoke',action='store_true')
    cli.add_argument('--freeze-only',action='store_true');arg=cli.parse_args()
    if arg.freeze_only:
        save(ROOT/'configs'/f'{arg.label}_stage{arg.stage}.json',configuration(arg.label,arg.stage)[-1]);return
    from reuse_cuda_cache import main as reuse_cache
    reuse_cache()
    if not arg.smoke:
        # Gate formal jobs even when a short run was launched before this wrapper revision.
        with (ROOT/'cache/smoke-replay.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            replay=ROOT/'smoke/N04/stage1/independent_replay.json'
            if not replay.exists():
                subprocess.run([sys.executable,'-u',str(ROOT/'scripts/smoke_validate.py'),'--label','N04'],check=True)
            assert json.loads(replay.read_text())['state']=='passed'
    torch.cuda.set_per_process_memory_fraction(.30)
    run_stage(arg.label,arg.stage,arg.smoke)
    if arg.smoke:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/smoke_validate.py'),'--label',arg.label],check=True)

if __name__=='__main__':main()
