from pathlib import Path
import json,subprocess
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parents[1];cases=json.loads((R/'nonvehicle10/inputs/cases.json').read_text());tasks=[]
for c in cases:tasks.append(([str(R/'env/bin/python'),str(R/'scripts/nonvehicle/compose.py'),'--case',c['case_id']],R/'nonvehicle10/logs'/f"{c['case_id']}_compose.log"))
for scene in ['scene-0103','scene-0104','scene-0277']:tasks.append(([str(R/'env/bin/python'),str(R/'scripts/compose_scene_validation.py'),'--scene',scene,'--case','random10_04_white_suv'],R/'scene_validation/logs'/f'{scene}_suv_compose.log'))
def run(t):
 cmd,log=t
 with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print(log.name,r.returncode,flush=True);return r.returncode
with ThreadPoolExecutor(max_workers=3) as p:result=list(p.map(run,tasks))
assert not any(result)
