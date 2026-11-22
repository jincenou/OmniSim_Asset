"""Two cold-start FTGS recipes frozen from their actual historical configs."""
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,random,sys,time
import numpy as np
import torch,yaml
from gpu_guard import validate_gpu
RUN=Path(__file__).resolve().parents[1];W=RUN.parents[1];OLD=W/'round2';sys.path[:0]=[str(RUN/'backend/src'),str(RUN/'backend')]
os.environ['TORCH_HOME']=str(RUN/'cache/torch');os.environ['MPLCONFIGDIR']=str(RUN/'cache/matplotlib')
import simple_trainer_freetime_4d_pure_relocation as module

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,data):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def main():
 parser=ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--branch',choices=['ftgs_base5k','ftgs_opt15k'],required=True);parser.add_argument('--resume',type=Path);arg=parser.parse_args();device=validate_gpu();sample=RUN/'samples'/arg.label;data=sample/'data';out=sample/arg.branch;frames=int(np.load(data/'init.npz')['frame_end']);assert frames in [61,121]
 historical=W/'reconstruction/outputs/phone_full/cfg.yml' if arg.branch=='ftgs_base5k' else OLD/'diagnostics/weak_velocity/zero_sigma8/cfg.yml';values=yaml.unsafe_load(historical.read_text());steps=5000 if arg.branch=='ftgs_base5k' else 15000
 if arg.resume:assert out.exists() and not (out/'training_complete.json').exists()
 else:out.mkdir(exist_ok=False)
 init=data/'init.npz'
 if arg.branch=='ftgs_opt15k':
  init=sample/'init_sigma8.npz'
  if not init.exists():
   with np.load(data/'init.npz') as z:arrays={k:z[k] for k in z.files}
   arrays['durations']=np.full_like(arrays['durations'],8/(frames-1));assert np.all(arrays['velocities']==0);np.savez_compressed(init,**arrays)
 values.update(data_dir=str(data),init_npz_path=str(init),result_dir=str(out),train_view_ids=[i for i in range(24) if i not in [0,8,16]],validation_view_ids=[0,8,16],end_frame=frames,render_traj_n_frames=frames,render_traj_time_frames=frames,max_steps=steps,lr_schedule_steps=steps,save_steps=[5000] if steps==5000 else [5000,10000,15000],eval_steps=[steps],ckpt_path=None)
 cfg=module.Config(**values);frozen=asdict(cfg);frozen_sha=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
 if arg.resume:
  original=json.loads((out/'experiment.json').read_text());assert original['frozen_config_sha256']==frozen_sha and original['init_sha256']==sha(init);cfg.ckpt_path=str(arg.resume)
 else:save(out/'experiment.json',{'branch':arg.branch,'full_config':frozen,'frozen_config_sha256':frozen_sha,'historical_config':str(historical),'historical_config_sha256':sha(historical),'init_sha256':sha(init),'base_init_sha256':sha(data/'init.npz'),'device':device,'zero_velocity':True,'initial_sigma_frames':15 if steps==5000 else 8,'initial_sigma_normalized':(15 if steps==5000 else 8)/(frames-1),'source_adaptation':'Only paths,clip length,split,fixedsave/evalschedule. B0own5kLRplan;B1cold15kplan. No capacity/loss/relocation parameter reselection.'})
 random.seed(42);np.random.seed(42);torch.manual_seed(42);start=time.monotonic();runner=module.FreeTime4DRunner(0,0,1,cfg);initial=len(runner.splats['means']);actual_sigma=runner.splats['durations'].detach().exp();expected_sigma=(15 if steps==5000 else 8)/(frames-1);assert torch.allclose(actual_sigma,torch.full_like(actual_sigma,expected_sigma),rtol=1e-5,atol=1e-7);assert torch.count_nonzero(runner.splats['velocities'])==0;save(out/'initial_parameters.json',{'point_count':initial,'duration_sigma_min':float(actual_sigma.min()),'duration_sigma_max':float(actual_sigma.max()),'expected_sigma':expected_sigma,'zero_velocity':True,'parameter_hashes':{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in runner.splats.items()},'camera_transform':runner.parser.transform.tolist()});base_tqdm=module.tqdm.tqdm
 class ObservedProgress(base_tqdm):
  def __iter__(self):
   for step in super().__iter__():
    yield step
    if self.total==steps and (step==0 or (step+1)%250==0):
     actual=runner.sampling_ledger.steps;assert all(torch.isfinite(v).all() for v in runner.splats.values()),'NonfiniteFTGSparams';record={'state':'training','step':step+1,'actual_optimizer_updates':actual,'gaussians':len(runner.splats['means']),'pid':os.getpid(),'device':device,'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()};save(out/'progress.json',record);print(json.dumps(record),flush=True)
 module.tqdm.tqdm=ObservedProgress
 try:runner.train()
 finally:module.tqdm.tqdm=base_tqdm
 assert runner.sampling_ledger.steps==steps;checkpoint=out/'ckpts'/f'ckpt_{steps-1}.pt';assert checkpoint.is_file() and (out/f'ckpt_{steps-1}.ftgs.ply').is_file()
 result={'state':'trained_exported_metrics_pending','steps':steps,'actual_optimizer_updates':runner.sampling_ledger.steps,'initial_gaussians':initial,'final_gaussians':len(runner.splats['means']),'elapsed_seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'device':device,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'sampling':runner.sampling_ledger.state()};save(out/'training_complete.json',result);print(json.dumps({k:v for k,v in result.items() if k!='sampling'}),flush=True)
if __name__=='__main__':main()
