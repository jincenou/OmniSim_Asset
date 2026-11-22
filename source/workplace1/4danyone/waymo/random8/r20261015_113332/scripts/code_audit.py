"""Bounded runtime trace of the existing time and coarse-stage conventions."""
from pathlib import Path
from argparse import ArgumentParser
from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,json,hashlib,random,sys,os,subprocess,gc
import numpy as np
import torch
RUN=Path(__file__).resolve().parents[1];OLD=RUN.parents[1]/'round2';A=OLD/'alternatives/4dgaussians';REPO=RUN/'source/4dgaussians'
sys.path.insert(0,str(REPO))
from scene import hexplane
from scene.deformation import deform_network
from scene.waymo_generated import GeneratedCameras

def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 physical=int(os.environ['CUDA_VISIBLE_DEVICES']);assert physical in [4,5,6,7]
 from gpu_guard import validate_gpu
 device=validate_gpu();uuid=device['cuda_uuid']
 observations=[]
 for label,baseline in [('m02_short61_24','m02_61_recipe30k_v1'),('phone_full','phone_recipe30k_v1')]:
  manifest=json.loads((A/'data'/label/'fdanyone.json').read_text());config=json.loads((A/'runs'/baseline/'experiment.json').read_text());frames=manifest['frame_count'];dataset=GeneratedCameras(manifest,[1]);indices=[0,frames//2,frames-1];cameras=[dataset[i] for i in indices];ts=torch.tensor([[c.time] for c in cameras],dtype=torch.float32)
  for size in [25,61]:
   hidden=copy.deepcopy(config['hidden']);hidden['kplanes_config']['resolution'][-1]=size;net=deform_network(SimpleNamespace(**hidden));grid=net.deformation_net.grid
   captured=[];original_sample=hexplane.grid_sample_wrapper
   def sample(tensor,coords,align_corners=True):
    captured.append({'shape':list(tensor.shape),'coordinates':coords.detach().tolist(),'align_corners':align_corners,'mode':'bilinear','padding_mode':'border'});return original_sample(tensor,coords,align_corners)
   hexplane.grid_sample_wrapper=sample
   grid_times=[];hook=grid.register_forward_pre_hook(lambda mod,args:grid_times.append(args[1].detach().tolist()))
   try:
    with torch.no_grad():net(torch.zeros((3,3)),torch.zeros((3,3)),torch.tensor([[1.,0,0,0]]*3),torch.zeros((3,1)),torch.zeros((3,16,3)),ts)
   finally:hexplane.grid_sample_wrapper=original_sample;hook.remove()
   assert grid_times==[ts.tolist()]
   planes=[captured[level*6+i] for level in range(len(hidden['multires'])) for i in [2,4,5]]
   assert all(p['shape'][2]==size and np.allclose(np.asarray(p['coordinates'])[:,1],ts.numpy()[:,0]) for p in planes)
   ramp=torch.arange(size,dtype=torch.float32).view(1,1,size,1).expand(1,1,size,2);coords=torch.cat([torch.zeros_like(ts),ts],1);actual=original_sample(ramp,coords).view(-1).tolist();want=[(float(t)+1)*(size-1)/2 for t in ts[:,0]];assert np.allclose(actual,want)
   observations.append({'label':label,'frames':frames,'dataset_times':ts[:,0].tolist(),'deformation_times':grid_times,'grid_time_size':size,'temporal_plane_shapes':[p['shape'] for p in planes],'sample_arguments':planes[0],'ramp_probe_node_indices':actual,'effective_time_node_range':[(size-1)/2,size-1],'effective_integer_nodes':(size+1)//2,'seconds_per_effective_node':manifest['sample_span_seconds']*2/(size-1),'spatial_transform':'normalize_aabb only forxyz;time concatenated unchanged','time_convention':'[0,1] reaches positivehalf ofgrid_sample[-1,1];inherited convention,not silently remapped'})
   del net,grid;gc.collect()
 # Actual short training uses old read-only generated inputs solely as an execution audit.
 import train
 from arguments import ModelParams,ModelHiddenParams,PipelineParams,OptimizationParams
 runs=[]
 for size in [25,61]:
  random.seed(42);np.random.seed(42);torch.manual_seed(42);parser=ArgumentParser();mp,hp,pp,op=ModelParams(parser),ModelHiddenParams(parser),PipelineParams(parser),OptimizationParams(parser);args=parser.parse_args([])
  args.source_path=str(A/'data/m02_short61_24');args.model_path=str(RUN/'research'/f'coarse_probe_attempt3_grid{size}');args.expname=f'random8_audit_{size}';args.data_device='cpu';args.render_process=False;args.coarse_iterations=3;args.iterations=3;args.kplanes_config=copy.deepcopy(args.kplanes_config);args.kplanes_config['resolution'][-1]=size
  train.args=args;calls=[];deform_calls=[];base_render=train.render
  def observed_render(camera,model,pipe,bg,*pos,**kw):
   stage=kw['stage'];calls.append({'stage':stage,'view':camera.source_view,'frame':camera.source_frame,'time':float(camera.time),'grad_enabled':torch.is_grad_enabled()})
   hook=model._deformation.register_forward_pre_hook(lambda m,a:deform_calls.append(stage))
   try:return base_render(camera,model,pipe,bg,*pos,**kw)
   finally:hook.remove()
  train.render=observed_render
  try:train.training(mp.extract(args),hp.extract(args),op.extract(args),pp.extract(args),[],[],[],None,-1,args.expname)
  finally:train.render=base_render
  optimized=[c for c in calls if c['grad_enabled']];assert all(c['view'] not in [0,8,16] for c in optimized);coarse=[c for c in optimized if c['stage']=='coarse'];fine=[c for c in optimized if c['stage']=='fine']
  assert len(coarse)==3 and len(fine)==3 and any(c['frame']!=0 for c in coarse);assert 'coarse' not in deform_calls and deform_calls.count('fine')==6
  sampling=json.loads((A/'runs/m02_61_recipe30k_v1/point_cloud/iteration_30000/sampling.json').read_text());counts=np.asarray(sampling['coarse']);assert counts[[0,8,16]].sum()==0
  runs.append({'grid_size':size,'configuration':{'dataloader':args.dataloader,'zerostamp_init':args.zerostamp_init},'actual_training_samples':optimized,'deformation_calls_by_stage':deform_calls,'historical_actual_coarse_3000':{'time_frames_seen':np.flatnonzero(counts.sum(0)>0).tolist(),'heldout_updates':int(counts[[0,8,16]].sum()),'updates':int(counts.sum())}})
  torch.cuda.empty_cache();gc.collect()
 assert runs[0]['actual_training_samples']==runs[1]['actual_training_samples']
 result={'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'state':'passed_inherited_time_and_coarse_conventions_risks_documented','hard_error':False,'source_pin':'843d5ac636c37e4b611242287754f3d4ed150144','device':{'physical_gpu':physical,'uuid':uuid,'pid':os.getpid()},'time_trace':observations,'coarse_trace':runs,'conclusions':['No duplicate/missing time conversion relative toinherited localrecipe;grid25/61 genuinelychanges temporalaxis','[0,1] uses13of25or31of61effective integernodes;do not reinterpret configured61as61independent effective steps','Coarse bypassesdeformation while samplingalltimes;zerostamp_initfalse,dataloaderfalse','Time0canonicalinitialization isnot coarse-only-time0supervision','No algorithmor time-range correction introduced;retain matched recipes','TrainingcameraHexcluded;initialization filter independent andalready source-hash audited'],'patches_this_audit':[]}
 save(RUN/'audit.json',result)
 text=['# 两项执行路径审查',f'更新时间：{result["updated_at"]}。真实CUDA审查通过，无需改时间范围或coarse监督规则。','## 时间','dataset时间frame/(N-1)直接进入camera.time、变形网络和HexPlane；xyz单独做AABB归一化，time未再变换。grid_sample为bilinear、align_corners=True、border padding。','时间[0,1]实际落在网格正半轴：25配置使用索引12到24（13个整数节点），61配置使用30到60（31个整数节点）。这是沿用配方的约定，本轮不擅自改为[-1,1]。61槽跨度2.4秒，对应约0.2/0.08秒每有效节点；121槽跨度4.8秒，对应0.4/0.16秒。','## Coarse','真实3步coarse＋3步fine的两种网格审查显示：coarse关闭形变，但抽样并不限首帧；既有3000步账本覆盖全部61时刻且H更新0。zerostamp_init=False、dataloader=False。第0时刻规范点云初始化不能解释成coarse只看首帧。两网格实际训练采样序列一致。','## 处理','上述风险记录但不增加消融或修改算法。独立源码副本保留原行为；本轮新分支均沿冻结配方。细节及实际张量/采样在audit.json。']
 (RUN/'code_audit.md').write_text('\n\n'.join(text)+'\n');print(json.dumps({'state':result['state'],'device':result['device']}),flush=True)
if __name__=='__main__':main()
