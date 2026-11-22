"""Continuous native trajectory with delayed ramped alpha; independent fusion branch."""
import importlib.util,sys,time,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;sys.path.insert(0,str(ROOT/'scripts'))
import train_grid as original
from grid_common import save,sha,render
begin=time.monotonic();spec=json.loads((NEW/'FUSION_PLAN.json').read_text())
assert sha(NEW/'FUSION_PLAN.json')==(NEW/'fusion_plan_hash.txt').read_text().strip()
assert sha(NEW/'source/train_fused.py')==spec['patched_source_sha256']
module_spec=importlib.util.spec_from_file_location('grid4d_fused_training',NEW/'source/train_fused.py');native=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(native);original.native=native
ParentScene=original.KnownScene
class AlphaScene(ParentScene):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        for cam,path in zip(self.train_cameras,self.loaded_paths):
            with Image.open(path) as im:cam.reference_alpha=torch.as_tensor(np.asarray(im)[...,3].copy(),device='cuda')
original.KnownScene=AlphaScene
ParentMonitor=original.Monitor
class FusionMonitor(ParentMonitor):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);native._fusion_alpha=self.alpha;self.alpha_value=0.;self.alpha_weight=0.
    def alpha(self,it,cam,gs,pipe,dx,dr,ds):
        self.alpha_weight=.1*min(1.,max(0.,(it-15000)/3000));self.alpha_value=0.
        if self.alpha_weight==0:return 0.
        target=cam.reference_alpha.float()/255;fg=target>.5;assert bool(fg.any()) and bool((~fg).any())
        alpha=render(cam,gs,pipe,torch.zeros(3,device='cuda'),dx,dr,ds,override_color=torch.ones_like(gs.get_xyz))['render'][0]
        error=(alpha-target).abs();loss=.5*(error[fg].mean()+error[~fg].mean());assert bool(torch.isfinite(loss));self.alpha_value=float(loss)
        return self.alpha_weight*loss
    def backward(self,*a,**kw):
        super().backward(*a,**kw)
        if self.alpha_weight>0:self.counts['training_renders']+=1
    def saved(self,it,scene,gs,deform):
        super().saved(it,scene,gs,deform)
        save(self.run/'budget'/f'{it}.json',dict(actual_updates=it-1,loop_seconds=self.wall_training,end_to_end_seconds=time.monotonic()-begin,alpha_weight=self.alpha_weight,render_calls=self.counts['training_renders']))
    def after(self,it,cam,gs,deform,loss,updated,stack):
        super().after(it,cam,gs,deform,loss,updated,stack)
        if it%100==0:save(self.run/'fusion_progress.json',dict(iteration=it,alpha_weight=self.alpha_weight,unweighted_alpha_loss=self.alpha_value,points=len(gs.get_xyz)))
original.Monitor=FusionMonitor
old_save=original.save
def save_cost(path,value):
    if Path(path).name=='complete.json':value.update(end_to_end_seconds=time.monotonic()-begin,fusion_plan_sha256=sha(NEW/'FUSION_PLAN.json'),actual_updates=32999)
    old_save(path,value)
original.save=save_cost
original.main()
