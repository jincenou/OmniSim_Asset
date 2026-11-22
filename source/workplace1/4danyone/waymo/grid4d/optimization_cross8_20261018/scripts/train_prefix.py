"""Pinned native33k trajectory, extra cost observations only."""
import sys,time,json
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent
sys.path.insert(0,str(ROOT/'scripts'))
import train_grid as original
from grid_common import save,sha
begin=time.monotonic()
Parent=original.Monitor
class CostMonitor(Parent):
    def saved(self,iteration,scene,gs,deform):
        start=time.monotonic()
        super().saved(iteration,scene,gs,deform)
        save(self.run/'budget'/f'{iteration}.json',dict(iteration=iteration,actual_updates=iteration-1,training_loop_wall_so_far=self.wall_training,end_to_end_seconds_so_far=time.monotonic()-begin,current_reference_save_seconds=time.monotonic()-start,scope='Native pre-update snapshot, full same training trajectory; H not used for stopping'))
original.Monitor=CostMonitor
old_save=original.save
def save_with_cost(path,value):
    if Path(path).name=='complete.json':
        value.update(end_to_end_seconds=time.monotonic()-begin,quality_plan_sha256=sha(NEW/'PLAN.json'))
    old_save(path,value)
original.save=save_with_cost
original.main()
