"""CPU audit of actual completed FTGS runs and foreground loss gradients."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1];W=ROOT.parent
BACKEND=W/'bridge/backend/src'
sys.path.insert(0,str(BACKEND))
from foreground_loss import composite_foreground,foreground_mse_loss

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_stats(a):
    a=np.asarray(a)
    if not np.issubdtype(a.dtype,np.number):
        return {'shape':list(a.shape),'dtype':str(a.dtype),'metadata':a.tolist()}
    return {'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'min':float(a.min()),'max':float(a.max())}

def asset(label):
    output=W/'reconstruction/outputs'/label
    # BaseLoader treats Python YAML tags as data and never constructs objects.
    cfg=yaml.load((output/'cfg.yml').read_text(),Loader=yaml.BaseLoader)
    keys=['alpha_mode','lambda_alpha','lambda_foreground','random_bkgd','lambda_img','lambda_ssim','lambda_perc',
       'position_lr','velocity_lr_start','velocity_lr_end','times_lr','durations_lr','auto_init_duration',
       'init_duration_multiplier','min_duration_frames','max_duration_frames','max_steps','lr_schedule_steps',
       'no_sampling','max_samples','use_keyframe_sampling','keyframe_step','use_relocation','use_pruning',
       'use_budget_pruning','relocation_every','relocation_stop_iter','densification_start_step','phases','strategy']
    init=Path(cfg['init_npz_path']);data=np.load(init)
    arrays={name:array_stats(data[name]) for name in data.files}
    checkpoint=output/'ckpts/ckpt_4999.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    splats=state['splats']
    final={name:array_stats(value.detach().numpy()) for name,value in splats.items()}
    metadata={key:value for key,value in state.items() if isinstance(value,(str,int,float,bool))}
    assert cfg['alpha_mode']=='transparent' and float(cfg['lambda_alpha'])>0
    assert cfg['random_bkgd']=='true' and cfg['phases']==[]
    assert all(a['finite'] for a in arrays.values() if 'finite' in a) and all(a['finite'] for a in final.values())
    return {'label':label,'config':{key:cfg.get(key) for key in keys},'config_sha256':sha(output/'cfg.yml'),
      'init_file':str(init),'init_sha256':sha(init),'init_arrays':arrays,'checkpoint':str(checkpoint),
      'checkpoint_sha256':sha(checkpoint),'checkpoint_scalar_metadata':metadata,'final_splats':final,
      'extra_foreground_loss_enabled':float(cfg['lambda_foreground'])>0,
      'interpretation':'Full-image RGB and alpha losses active; extra foreground MSE disabled. Strategy refinement starts beyond this run; relocation does not create extra capacity.'}

def gradient_case(background):
    alpha=torch.tensor([[[[.7],[.6]]]],dtype=torch.float64,requires_grad=True)
    # First pixel is an opaque black floater OUTSIDE target mask.
    rgb=torch.tensor([[[[0.,0.,0.],[.12,.24,.36]]]],dtype=torch.float64)
    target_rgb=torch.tensor([[[[.9,.1,.7],[.2,.4,.6]]]],dtype=torch.float64)
    target_alpha=torch.tensor([[[[0.],[1.]]]],dtype=torch.float64)
    bg=torch.full((1,3),background,dtype=torch.float64)
    pred,target,alpha_loss=composite_foreground(rgb,alpha,target_rgb,target_alpha,bg)
    assert torch.allclose(target[0,0,0],bg[0])
    assert torch.allclose(pred[0,0,1],rgb[0,0,1]+.4*bg[0])
    loss=.8*(pred.clamp(0,1)-target).abs().mean()+.1*alpha_loss
    loss.backward();gradient=float(alpha.grad[0,0,0,0]);assert gradient>0
    extra=foreground_mse_loss(pred,target,target_alpha)
    return {'background':background,'outside_black_floater_alpha_gradient':gradient,
       'positive_gradient_reduces_alpha_under_gradient_descent':True,'loss':float(loss.detach()),'optional_foreground_mse':float(extra.detach()),
       'scope':'Real composition/loss functions on CPU; not a CUDA rasterizer gradient or full training quality test'}

result={'state':'completed_cpu_audit','gpu_used':False,
 'source_hashes':{str(p):sha(p) for p in [BACKEND/'foreground_loss.py',BACKEND/'simple_trainer_freetime_4d_pure_relocation.py']},
 'assets':list(map(asset,['phone_full','pilot_043_twentyfour'])),
 'gradient_checks':list(map(gradient_case,[0.,.5,1.])),
 'next_actions':['Retain Q0 loss recipe for budget/resolution controls; changing foreground weight is a separate experiment',
   'Audit masks and train/heldout render errors before increasing capacity',
   'Instrument actual per-view/time samples in isolated round2 trainer; historical replay alone is not actual sample logging',
   'Run full CUDA rasterizer exterior-alpha gradient check after batch24 completes']}
(ROOT/'quality_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'state':result['state'],'assets':[a['label'] for a in result['assets']],
 'foreground_loss_enabled':[a['extra_foreground_loss_enabled'] for a in result['assets']],
 'gradient_checks_passed':len(result['gradient_checks'])}),flush=True)
