import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from hasim.shared.assets.library_v2 import AssetLibrary
from hasim.shared.insertion import InsertionScenario
from hasim.shared.trajectory_file import Track,TrajectoryFile
B=Path('/mnt/ojc/workplace1/human_cycle_update_20261108')
def one(identity):
 lib=AssetLibrary('/mnt/ojc/workplace1/asset_library/human');a=lib.get(identity);p=a.meta['cycle_profile'];speed=max(.050001,p['native_speed_mps']);track=Track('person','human',[0.,30.],[[0.,0.,0.],[30*speed,0.,0.]],[0.,0.]);s=InsertionScenario(TrajectoryFile((track,)),lib,bindings={'person':identity},playback={'default':'cycle'})
 for t in [0.,15.,29.]:assert len(s.sample(t).instances)==1
 return {'id':identity,'passed':True,'times':[0,15,29]}
if __name__=='__main__':
 ids=[r['id'] for r in json.loads((B/'optimization_trial/summary.json').read_text())['results'] if r['rescued']]
 with ProcessPoolExecutor(4) as p:rows=list(p.map(one,ids))
 (B/'v2/added_binding_validation.json').write_text(json.dumps(rows,indent=2));print(len(rows),'passed',flush=True)
