"""Explicit sampled RGB+alpha optimization of the author's native dynamic4D Gaussians."""
import argparse,json,time,sys,traceback,hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--object',choices=['R07','M02']);p.add_argument('--protocol',default='S1');p.add_argument('--manifest',type=Path);p.add_argument('--variant',choices=['E0','E1'],default='E0');p.add_argument('--run-id');p.add_argument('--steps',type=int);p.add_argument('--accumulation',type=int);p.add_argument('--replay',type=Path);args=p.parse_args()
import numpy as np,torch
from PIL import Image
from runtime import *
from data import output_path,model_time,observation_seconds

def hash_parameters(model):
 fields=['_xyz','_t','_scaling','_scaling_t','_rotation','_rotation_r','_opacity','_features_dc','_features_rest']
 return {f:hashlib.sha256(getattr(model,f).detach().cpu().contiguous().numpy().tobytes()).hexdigest() for f in fields}

def replay(path):
 model,d=load(path);ref=torch.load(path.parent/'replay_reference.pt',weights_only=False);rows=[]
 with torch.no_grad():
  for query,target in zip(ref['queries'],ref['images']):
   o=render(model,query['camera'],query['time']);got=torch.cat([o['render'],o['alpha']]).cpu();err=float((got-target).abs().max());assert err<1e-6
   rows.append({'camera':query['camera']['id'],'time':query['time'],'max_abs':err})
 save_json(path.parent/'replay.json',{'state':'passed','device':device_record(),'new_process':True,'checkpoint_sha256':sha(path),'queries':rows})

if args.replay:replay(args.replay);sys.exit()
if (not args.object and not args.manifest) or not args.run_id or not args.steps:p.error('object/manifest, run-id and steps are required')
assert args.steps>0
assert json.loads((ROOT/'research/preflight/result.json').read_text())['state']=='passed'
manifest=args.manifest if args.manifest else ROOT/'objects'/({'R07':'OBJ_0018','M02':'OBJ_0019'}[args.object])/'clips/C001/manifest.json';m=json.loads(manifest.read_text());prot=dict(m['protocols'][args.protocol])
args.accumulation=prot['gradient_accumulation'] if args.accumulation is None else args.accumulation
assert args.accumulation>0
prot['gradient_accumulation']=args.accumulation
out=output_path(manifest.parent/'05_reconstruction'/args.run_id/'config.json').parent
if (out/'config.json').exists():raise FileExistsError(out)
device=device_record();start=time.monotonic();conf=None
for path,h in m['source_hashes'].items():assert sha(path)==h,path
init=manifest.parent/'05_reconstruction'/f'init_{args.protocol}.npz'
config={'object':args.object,'protocol':args.protocol,'variant':args.variant,'run_id':args.run_id,'steps':args.steps,'accumulation':args.accumulation,'microbatch':1,'views_per_microbatch':1,'times_per_microbatch':1,'seed_initialization':20261014,'seed_sampling':20261015,'manifest':str(manifest),'manifest_sha256':sha(manifest),'init':str(init),'init_sha256':sha(init),'sampling':prot,'initial_spatial_sigma':.022,'initial_temporal_sigma_model':.15,'densification':False,'pruning':False,'background':.5,'rgb_loss':'weighted MSE / (3*sum weights), E0 weights=1','alpha_loss':'full-image unweighted MSE coefficient1','lambda_sd':0,'lambda_zero123':0,'lambda_svd':0,'diffusion_weights_loaded':False,'max_wall_seconds':7200,'lr':vars(arguments(args.steps)),'source_commit':'35b67b062d60ca5a28f6a5fab677a2e0b4d5cb87','device':device}
if args.variant=='E1':
 cp=manifest.parent/'05_reconstruction'/f'confidence_{args.protocol}'/'summary.json';summary=json.loads(cp.read_text());assert summary['state']=='complete';assert summary['manifest_sha256']==sha(manifest)
 config['confidence_summary_sha256']=sha(cp);conf={}
 for v in summary['views']:
  assert sha(v['path'])==v['sha256'];d=np.load(v['path']);assert np.array_equal(d['frames'],prot['train_frame_ids']);conf[v['view']]=d['weights'].copy()
save_json(out/'config.json',config)
try:
 cams={c['id']:c for c in m['cameras']};frames=prot['train_frame_ids'];samples=[(v,f) for v in prot['train_camera_ids'] for f in frames]
 # CPU uint8 observations. One image graph at a time; no entire sequence GPU graph.
 images={}
 for v,f in samples:
  path=Path(m['data_root'])/'images'/f'cam{v:02d}'/f'{f:06d}.png';images[(v,f)]=np.array(Image.open(path))
 model=create(init,args.steps,config['seed_initialization']);initial=hash_parameters(model);save_json(out/'initial_parameter_hashes.json',initial)
 rng=np.random.default_rng(config['seed_sampling']);checkpoint(out/'initial.pt',model,config,0,rng)
 order=np.empty(0,dtype=int);cursor=0;counts=np.zeros(len(samples),dtype=np.int32);history=[];sample_digest=hashlib.sha256();grad_test=None
 torch.cuda.reset_peak_memory_stats();training_start=time.monotonic()
 for step in range(args.steps):
  if time.monotonic()-start>config['max_wall_seconds']:raise TimeoutError('Run exceeded fixed wall-clock budget')
  model.update_learning_rate(step);model.optimizer.zero_grad(set_to_none=True);total=0.;parts={'rgb':0.,'alpha':0.}
  for _ in range(args.accumulation):
   if cursor==len(order):order=rng.permutation(len(samples));cursor=0
   index=int(order[cursor]);cursor+=1;counts[index]+=1;sample_digest.update(index.to_bytes(4,'little'));v,f=samples[index]
   tensor=torch.from_numpy(images[(v,f)]).to(device='cuda',dtype=torch.float32).permute(2,0,1)/255;rgb=tensor[:3];alpha=tensor[3:]
   weight=None if conf is None else torch.from_numpy(conf[v][frames.index(f)].astype('float32')).to('cuda')[None]
   pred=render(model,cams[v],model_time(m,observation_seconds(m,f)));loss,components=loss_image(pred,rgb,alpha,weight)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
   if weight is not None and grad_test is None and f not in [frames[0],frames[-1]]:
    equal,_=loss_image(pred,rgb,alpha);g0=torch.autograd.grad(equal,model._features_dc,retain_graph=True)[0];g1=torch.autograd.grad(loss,model._features_dc,retain_graph=True)[0]
    delta=float((g1-g0).abs().max());assert delta>1e-10
    grad_test={'view':v,'frame':f,'equal_loss':float(equal.detach()),'weighted_loss':float(loss.detach()),'feature_gradient_max_change':delta,'confidence_detached':not weight.requires_grad};save_json(out/'confidence_loss_gradient_test.json',grad_test)
   (loss/args.accumulation).backward();total+=float(loss.detach())/args.accumulation
   for k in parts:parts[k]+=components[k]/args.accumulation
  for group in model.optimizer.param_groups:
   param=group['params'][0]
   if param.grad is not None and not torch.isfinite(param.grad).all():raise FloatingPointError(f"Nonfinite gradient {group['name']}")
  model.optimizer.step()
  if step%10==0 or step+1==args.steps:
   torch.cuda.synchronize();row={'step':step+1,'loss':total,**parts,'rendered_samples':(step+1)*args.accumulation,'training_seconds':time.monotonic()-training_start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()};history.append(row)
   save_json(out/'progress.json',{'state':'training','device':device,**row});print(json.dumps(row),flush=True)
  if args.steps>150 and (step+1)%max(1,args.steps//3)==0:checkpoint(out/f'checkpoint_{step+1:06d}.pt',model,config,step+1,rng,{'order':order,'cursor':cursor,'counts':counts,'sample_digest':sample_digest.hexdigest()})
 training_seconds=time.monotonic()-training_start
 checkpoint(out/'checkpoint.pt',model,config,args.steps,rng,{'order':order,'cursor':cursor,'counts':counts,'sample_digest':sample_digest.hexdigest()})
 assert hash_parameters(model)!=initial
 queries=[{'camera':cams[v],'time':t} for v in [0,8,16] for t in [0,.173,.5,1]];replay_images=[]
 with torch.no_grad():
  for i,q in enumerate(queries):
   result=render(model,q['camera'],q['time']);replay_images.append(torch.cat([result['render'],result['alpha']]).cpu())
   if q['camera']['id']==8:
    rgb=(result['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8');Image.fromarray(rgb).save(out/f"preview_view08_t{q['time']:.3f}.png")
 torch.save({'queries':queries,'images':replay_images},out/'replay_reference.pt')
 import subprocess
 subprocess.run([sys.executable,__file__,'--replay',str(out/'checkpoint.pt')],check=True,cwd=ROOT)
 for path,h in m['source_hashes'].items():assert sha(path)==h,path
 if args.variant=='E1':assert grad_test is not None
 save_json(out/'complete.json',{'state':'trained_and_independent_replay_passed_evaluation_pending','device':device,'steps':args.steps,'rendered_samples':args.steps*args.accumulation,'training_seconds':training_seconds,'total_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'points':len(model._xyz),'initial_parameter_hashes':initial,'final_parameter_hashes':hash_parameters(model),'sample_sequence_sha256':sample_digest.hexdigest(),'observation_counts':[{'camera':v,'frame':f,'count':int(n)} for (v,f),n in zip(samples,counts)],'history':history,'source_hashes_rechecked':True,'confidence_gradient_test':grad_test,'production_approved':False})
except BaseException as e:
 save_json(out/'failure.json',{'state':'failed','exception':str(e),'traceback':traceback.format_exc(),'device':device});raise
