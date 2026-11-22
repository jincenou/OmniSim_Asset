"""CPU-only released-weight/sequence-length check; not a motion-quality test."""
from pathlib import Path
from types import SimpleNamespace
import json,sys
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/RoHM'))
from utils.other_utils import REPR_DIM_DICT
from model.posenet import PoseNet
from model.trajnet import TrajNet

def main():
 torch.set_num_threads(2)
 checkpoints=ROOT/'models/rohm/checkpoints';device=torch.device('cpu')
 body=sum(REPR_DIM_DICT.values());pose=sum(REPR_DIM_DICT[k] for k in ['local_positions','local_vel','smplx_body_pose_6d','smplx_betas','foot_contact'])
 dataset=SimpleNamespace(body_feat_dim=body,pose_feat_dim=pose,traj_feat_dim=body-pose)
 model=PoseNet(dataset=dataset,body_feat_dim=body,latent_dim=512,ff_size=1024,num_layers=8,num_heads=4,dropout=.1,activation='gelu',
  body_model_path=str(ROOT.parent.parent/'repo/models/body_models'),device=device,traj_feat_dim=dataset.traj_feat_dim)
 weights=torch.load(checkpoints/'posenet_checkpoint/model000200000.pt',map_location='cpu',weights_only=True)
 model.load_state_dict(weights,strict=True);del weights;model.eval()
 report={'device':'cpu','scope':'Strict loading and finite single forward, not sampling, geometry recovery or quality validation','PoseNet':{'strict_loaded':True,'lengths':{}}}
 with torch.inference_mode():
  for length in [72,144]:
   batch={'x_t':torch.zeros(1,body,1,length),'cond':torch.zeros(1,body,1,length)}
   y=model(batch,torch.tensor([1]));assert y.shape==(1,body,1,length) and torch.isfinite(y).all()
   report['PoseNet']['lengths'][str(length)]={'state':'finite_forward','shape':list(y.shape)}
 del model
 for control,name in [(False,'trajnet'),(True,'trajnet_control')]:
  traj=TrajNet(time_dim=32,mid_dim=512,cond_dim=13,traj_feat_dim=13,trajcontrol=control,device=device,dataset=SimpleNamespace(traj_feat_dim=13),repr_abs_only=True)
  file=checkpoints/(name+'_checkpoint')/('model000400000.pt' if control else 'model000450000.pt')
  traj.load_state_dict(torch.load(file,map_location='cpu',weights_only=True),strict=True);traj.eval()
  report[name]={'strict_loaded':True,'lengths':{}}
  with torch.inference_mode():
   for length in [144,72]:
    try:
     batch={'x_t':torch.zeros(1,length,13),'cond':torch.zeros(1,length,13),'control_cond':torch.zeros(1,length,pose)}
     y=traj(batch,torch.tensor([1]));assert torch.isfinite(y).all()
     report[name]['lengths'][str(length)]={'state':'finite_forward','shape':list(y.shape)}
    except (RuntimeError,ValueError) as e:
     report[name]['lengths'][str(length)]={'state':'unsupported_or_contract_mismatch','error':str(e)}
  del traj
 (ROOT/'rohm_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
