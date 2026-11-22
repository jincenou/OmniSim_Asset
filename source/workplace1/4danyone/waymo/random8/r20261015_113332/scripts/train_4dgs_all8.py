"""Frozen local HUST recipe,independent sampling RNG and lossless resume records."""
from argparse import ArgumentParser
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import copy,gzip,hashlib,json,os,random,sys,time
import numpy as np
import torch
from all8_gpu_guard import validate_gpu
RUN=Path(__file__).resolve().parents[1];OLD=RUN.parents[1]/'round2';A=RUN/'four_dg';REPO=RUN/'source/4dgaussians';sys.path.insert(0,str(REPO))
os.environ['TORCH_HOME']=str(RUN/'cache/torch');os.environ['MPLCONFIGDIR']=str(RUN/'cache/matplotlib')
import train
from arguments import ModelParams,ModelHiddenParams,PipelineParams,OptimizationParams

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def save(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
 cli=ArgumentParser();cli.add_argument('--label',required=True);cli.add_argument('--grid',type=int,choices=[25,61],required=True);cli.add_argument('--resume',type=Path);arg=cli.parse_args();device=validate_gpu();assert json.loads((RUN/'audit.json').read_text())['hard_error'] is False
 manifest=A/'data'/arg.label/'fdanyone.json';data=json.loads(manifest.read_text());out=A/'runs'/f'{arg.label}_grid{arg.grid}';base=json.loads((OLD/'alternatives/4dgaussians/runs/m02_61_recipe30k_v1/experiment.json').read_text())
 parser=ArgumentParser();mp,hp,pp,op=ModelParams(parser),ModelHiddenParams(parser),PipelineParams(parser),OptimizationParams(parser);args=parser.parse_args([])
 for section in ['dataset','hidden','pipeline','optimization']:
  for key,value in base[section].items():setattr(args,key,copy.deepcopy(value))
 args.source_path=str(manifest.parent);args.model_path=str(out);args.expname=f'{arg.label}_grid{arg.grid}';args.render_process=False;args.data_device='cpu';args.kplanes_config['resolution'][-1]=arg.grid;args.coarse_iterations=3000;args.iterations=30000
 random.seed(42);np.random.seed(42);torch.manual_seed(42);sampler=random.Random(42);train.args=args;train.network_gui.listener.close()
 config={'dataset':vars(mp.extract(args)),'hidden':vars(hp.extract(args)),'pipeline':vars(pp.extract(args)),'optimization':vars(op.extract(args)),'seed':42,'sampling_seed':42,'objective':'official_rgb','source_commit':'843d5ac636c37e4b611242287754f3d4ed150144','trainer_sha256':sha(REPO/'train.py'),'device':device,'input_manifest_sha256':sha(manifest),'started_at':stamp(),'scope':'Frozen3kcoarse+30kfine;grid25/61onlyexplicitdifference;independentPython samplingRNG;gzipcheckpoint lossless,not quantized.','budget_note':'fine30000backwarditerations;original optimizer predicate skips lastfine update =>32999actual optimizer steps.','snapshot_note':'native pointcloud saves beforeiteration optimizer step;resumecheckpoint saves afterstep;fine12000 is nominal total15k.'}
 frozen={k:v for k,v in config.items() if k not in ['device','started_at']};config['frozen_config_sha256']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
 if arg.resume:
  assert out.is_dir();previous=json.loads((out/'experiment.json').read_text());assert previous['frozen_config_sha256']==config['frozen_config_sha256'];assert arg.resume.is_file()
 else:out.mkdir(parents=True,exist_ok=False);save(out/'experiment.json',config)
 active={};trace=[];actual_updates={'coarse':0,'fine':0};initial=[None];original_stage=train.scene_reconstruction;original_randint=train.randint;original_render=train.render;original_save=torch.save;original_load=torch.load
 def stage(dataset,opt,hyper,pipe,testing,saving,checkpoints,checkpoint,debug,model,scene,name,*rest):
  active.update(scene=scene,model=model,stage=name,remaining=list(range(len(scene.getTrainCameras()))));initial[0]=initial[0] or len(model.get_xyz)
  return original_stage(dataset,opt,hyper,pipe,testing,saving,checkpoints,checkpoint,debug,model,scene,name,*rest)
 def randint(lo,hi):
  assert lo==0 and hi+1==len(active['remaining']);value=sampler.randint(lo,hi);index=active['remaining'].pop(value);camera=active['scene'].getTrainCameras()[index];assert camera.source_view not in [0,8,16];trace.append([active['stage'],camera.source_view,camera.source_frame])
  if not active['remaining']:active['remaining']=list(range(len(active['scene'].getTrainCameras())))
  return value
 def after_step(scene,name,iteration,model):
  actual_updates[name]+=1
  if iteration==1 or iteration%250==0:
   progress={'state':'training','stage':name,'stage_iteration':iteration,'global_nominal_iteration':iteration+(3000 if name=='fine' else 0),'optimizer_updates':dict(actual_updates),'actual_optimizer_updates':sum(actual_updates.values()),'gaussians':len(model.get_xyz),'pid':os.getpid(),'gpu':device,'updated_at':stamp()};save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
 train._random8_after_step=after_step
 def save_checkpoint(payload,path,*pos,**kwargs):
  path=Path(path)
  if path.name.startswith('chkpnt_'):
   target=path.with_suffix(path.suffix+'.gz');temp=target.with_suffix(target.suffix+'.tmp');scene=active['scene'];model=active['model']
   runtime={'sampler_rng':sampler.getstate(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'remaining':active['remaining'],'trace':trace,'optimizer_updates':actual_updates,'sampling':scene.waymo_sampling,'stage':active['stage'],'deformation_accum':model._deformation_accum,'config_sha256':config['frozen_config_sha256'],'input_manifest_sha256':sha(manifest)}
   with gzip.open(temp,'wb',compresslevel=1) as file:original_save({'core':payload,'runtime':runtime},file)
   temp.replace(target);save(target.with_suffix('.json'),{'stage':runtime['stage'],'stage_iteration':payload[1],'global_nominal_iteration':payload[1]+(3000 if runtime['stage']=='fine' else 0),'actual_optimizer_updates':sum(actual_updates.values()),'sha256':sha(target),'config_sha256':runtime['config_sha256'],'input_manifest_sha256':runtime['input_manifest_sha256'],'compression':'losslessgzip'})
  else:return original_save(payload,str(path),*pos,**kwargs)
 def load_checkpoint(path,*pos,**kwargs):
  if isinstance(path,(str,Path)) and str(path).endswith('.pth.gz'):
   proof=json.loads(Path(path).with_suffix('.json').read_text());assert sha(Path(path))==proof['sha256']
   with gzip.open(path,'rb') as file:payload=original_load(file,weights_only=False)
   runtime=payload['runtime'];assert runtime['config_sha256']==config['frozen_config_sha256'] and runtime['input_manifest_sha256']==sha(manifest);active['resume']=runtime;return payload['core']
  return original_load(path,*pos,**kwargs)
 def restore_runtime(scene,name,temp_list):
  if 'resume' not in active or active['resume']['stage']!=name:return None
  state=active['resume'];sampler.setstate(state['sampler_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);active['remaining']=state['remaining'];trace[:]=state['trace'];actual_updates.update(state['optimizer_updates']);scene.waymo_sampling=state['sampling'];active['model']._deformation_accum=state['deformation_accum'];return [temp_list[i] for i in active['remaining']]
 train._random8_restore_runtime=restore_runtime;train.scene_reconstruction=stage;train.randint=randint;torch.save=save_checkpoint;torch.load=load_checkpoint
 started=time.monotonic()
 try:train.training(mp.extract(args),hp.extract(args),op.extract(args),pp.extract(args),[],[3000,12000,30000],[3000,12000,30000],str(arg.resume) if arg.resume else None,-1,args.expname)
 finally:torch.save=original_save;torch.load=original_load;train.scene_reconstruction=original_stage;train.randint=original_randint
 assert actual_updates=={'coarse':3000,'fine':29999},actual_updates
 final=out/'point_cloud/iteration_30000';sampling=json.loads((final/'sampling.json').read_text());assert all(sum(sampling[stage][i])==0 for stage in ['coarse','fine'] for i in [0,8,16]);assert sum(map(sum,sampling['coarse']))==3000 and sum(map(sum,sampling['fine']))==30000
 schedule=np.asarray([[0 if x[0]=='coarse' else 1,x[1],x[2]] for x in trace],dtype=np.uint16);np.save(out/'sampling_trace.npy',schedule)
 assert len(trace)==33000
 complete={'state':'trained_exported_replay_and_metrics_pending','device':device,'coarse_iterations':3000,'fine_iterations':30000,'actual_optimizer_updates':sum(actual_updates.values()),'actual_updates_by_stage':actual_updates,'elapsed_seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'initial_gaussians':initial[0],'final_gaussians':len(active['model'].get_xyz),'network_parameters':sum(p.numel() for p in active['model']._deformation.parameters()),'sampling_trace_sha256':sha(out/'sampling_trace.npy'),'ended_at':stamp()};save(out/'complete.json',complete);print(json.dumps(complete),flush=True)
if __name__=='__main__':main()
