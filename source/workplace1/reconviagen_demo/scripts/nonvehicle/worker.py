from pathlib import Path
import os,sys,json,subprocess,argparse,time,traceback
R=Path(__file__).resolve().parents[2];O=R/'nonvehicle10';p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--cases',nargs='+',required=True);a=p.parse_args()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(R/'cache/matplotlib'),HF_HOME=str(R/'cache/hf'),XDG_CACHE_HOME=str(R/'cache/xdg'),PYTHONUNBUFFERED='1',TORCH_HOME='/mnt/ojc/models/trellis_runtime/torch',NUMBA_CACHE_DIR=str(R/'cache/numba'),PATH='/home/wmj/miniconda3/envs/sim/bin:'+env['PATH'],CPATH='/home/wmj/miniconda3/envs/splatad/include',TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),MAX_JOBS='4')
def run(cmd,log):
 with log.open('w') as f:r=subprocess.run([str(x) for x in cmd],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 log.with_suffix('.exit').write_text(str(r.returncode))
 if r.returncode:raise RuntimeError(str(log))
for cid in a.cases:
 manifest=O/'inputs'/cid/'segmented_manifest.json';base=json.loads((manifest.parent/'manifest.json').read_text());d=json.loads(manifest.read_text());allowed={r['id'] for r in base['images']};d['images']=[r for r in d['images'] if r['id'] in allowed];manifest.write_text(json.dumps(d,indent=2))
 assert all(Path(r['rgba_path']).exists() for r in d['images'])
 for v in ['B0','R','H','B0_real1']:
  out=O/'results'/cid/f'{v}_seed42';out.mkdir(parents=True,exist_ok=True);tag=f'{cid}_{v}';print('START',tag,flush=True)
  try:
   if not (out/'inference.json').exists():
    if v.startswith('B0'):cmd=[R/'env/bin/python',R/'scripts/nonvehicle/baseline.py','--case',cid,'--real-count','1' if v=='B0_real1' else '0']
    else:cmd=[R/'env/bin/python',R/'scripts/nonvehicle/run_case.py','--case',cid,'--variant',v]
    run(cmd,O/'logs'/f'{tag}_generate.log')
   if not (out/'asset/metadata.json').exists():run([R/'env/bin/python',R/'scripts/nonvehicle/prepare_asset.py','--case',cid,'--variant',v],O/'logs'/f'{tag}_export.log')
   if not (out/'orbit/render_config.json').exists():run(['/home/wmj/miniconda3/envs/sim/bin/python',R/'scripts/nonvehicle/render_orbit.py','--ply',out/'asset/canonical.ply','--output',out/'orbit'],O/'logs'/f'{tag}_render.log')
   print('DONE',tag,flush=True)
  except Exception:
   (out/'worker_failure.txt').write_text(traceback.format_exc());print('FAILED',tag,flush=True)
for enc in ['clip','dino']:run(['/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python',R/'scripts/nonvehicle/evaluate_features.py','--encoder',enc,'--cases',*a.cases],O/'logs'/f'gpu{a.gpu}_{enc}.log')
run([R/'env_refine/bin/python',R/'scripts/nonvehicle/evaluate_pixels.py','--cases',*a.cases],O/'logs'/f'gpu{a.gpu}_pixels.log');print('WORKER COMPLETE',a.gpu,flush=True)
