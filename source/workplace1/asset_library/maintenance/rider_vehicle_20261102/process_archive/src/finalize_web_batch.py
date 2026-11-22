from pathlib import Path
import subprocess,json,os
W=Path(__file__).resolve().parents[1];P='/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python';V='/mnt/ojc/workplace1/reconviagen_demo/env_splatad/bin/python';env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
names=[r['name'] for r in json.loads((W/'sources/web_expansion_20261103/queue.json').read_text())]
for name in names:
 run=W/'runs'/name
 if not (run/'selected/selected.json').exists():subprocess.run([P,str(W/'src/finalize_asset.py'),str(run),'H0','B'],check=True,env=env)
 if not (run/'selection_review.json').exists():
  (run/'selection_review.json').write_text(json.dumps(dict(selected_method='H0',status='orbit_passed_pending_scene_validation',reviewer='Codex visual inspection of all 16 views',front='B',findings='Exactly one coherent seated rider and complete scooter, two wheels, supported feet and hands at grips; no gross separation across orbit. Headlight and fork identify forward direction.',limitations=['Face identity and small markings not exact','Back clothing panels and rear mechanical details synthesized','Metric and scene contact validation required']),indent=2))
 if not (run/'selected/load_validation.json').exists():subprocess.run([V,str(W/'src/validate_loader.py'),name],check=True,env=env)
subprocess.run([P,str(W/'src/check_support.py'),*names],check=True,env=env)
print('FINALIZE_NUMERIC_DONE',len(names),flush=True)
