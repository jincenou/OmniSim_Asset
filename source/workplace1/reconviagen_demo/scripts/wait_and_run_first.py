"""Bounded first-object jobs: wait for verified checkpoint rename and free GPU."""
from pathlib import Path
import argparse,os,subprocess,time,json,traceback
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['R','F','H'],required=True)
p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
weight=ROOT/'cache/Stable-X--vggt-object-v0-1/model.safetensors'
deadline=time.monotonic()+7200
print('Waiting for verified VGGT weight and idle GPU',a.gpu,flush=True)
while True:
    if time.monotonic()>deadline:raise TimeoutError('Weight/GPU wait exceeded 2 hours')
    if weight.exists():
        memory=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory<100:break
    time.sleep(10)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1')
print('Starting first-object variant',a.variant,flush=True)
subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/run_case.py'),'--case','service_van','--variant',a.variant],env=env,check=True)
subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/prepare_asset.py'),'--case','service_van','--variant',a.variant],env=env,check=True)
case=json.loads((ROOT/'inputs/service_van/manifest.json').read_text())
env.update(PATH='/home/wmj/miniconda3/envs/sim/bin:'+env['PATH'],CPATH='/home/wmj/miniconda3/envs/splatad/include',
    TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),MAX_JOBS='4')
out=ROOT/'results/service_van'/f'{a.variant}_seed42/asset'
subprocess.run(['/home/wmj/miniconda3/envs/sim/bin/python',str(ROOT/'scripts/render_orbit.py'),
    '--ply',str(out/'canonical.ply'),'--output',str(out),'--length',str(case['metric_dimensions_lwh'][0]),'--distance','11'],env=env,check=True)
print('Inference, canonical reload, and orbit complete; visual heading and scene insertion pending',flush=True)
