import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from hasim.shared.assets.library_v2 import AssetLibrary
from hasim.shared.insertion import InsertionScenario
from hasim.shared.trajectory_file import Track,TrajectoryFile
B=Path('/mnt/ojc/workplace1/human_cycle_update_20261108')
def run(identity):
 lib=AssetLibrary('/mnt/ojc/workplace1/asset_library/human');asset=lib.get(identity);assert asset.meta['cycle_profile']['mode']=='stationary'
 t=Track('person','human',[0.,30.],[[0.,0.,0.],[0.,0.,0.]],[0.,0.]);s=InsertionScenario(TrajectoryFile((t,)),lib,bindings={'person':identity},playback={'default':'cycle'})
 for time in [0.,15.,29.]:assert len(s.sample(time).instances)==1
 return {'id':identity,'passed':True,'times':[0.,15.,29.],'trajectory':'stationary'}
if __name__=='__main__':
 rows=[json.loads(p.read_text()) for p in (B/'classification_review/cycle_results/reports').glob('*.json')];ids=[r['id'] for r in rows if r['status']=='qualified_numeric']
 with ProcessPoolExecutor(8) as p:out=list(p.map(run,ids))
 (B/'v3/classified_runtime_validation.json').write_text(json.dumps(out,indent=2));print(len(out),'passed',flush=True)
