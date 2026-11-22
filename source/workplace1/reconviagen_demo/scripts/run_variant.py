"""One GPU: generation, original-loader export, then unified orbit rendering."""
from pathlib import Path
import os,json,argparse,subprocess
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',choices=['B0','R','F','H','B0_real1','B0_real4','B0_repeat','R1','R2'],required=True);p.add_argument('--gpu',required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
args=['--case',a.case,'--seed',str(a.seed)]
script='baseline.py' if a.variant.startswith('B0') else 'run_case.py'
if a.variant.startswith('B0_real'):args+=['--real-count',a.variant[-1]]
elif a.variant=='B0_repeat':args+=['--repeat-control']
elif a.variant in ['R1','R2']:args+=['--variant','R','--real-count',a.variant[-1]]
elif a.variant!='B0':args+=['--variant',a.variant]
subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts'/script)]+args,env=env,check=True)
subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/prepare_asset.py'),'--case',a.case,'--variant',a.variant,'--seed',str(a.seed)],env=env,check=True)
case=json.loads((ROOT/'inputs'/a.case/'manifest.json').read_text());out=ROOT/'results'/a.case/f'{a.variant}_seed{a.seed}'/'asset'
env.update(PATH='/home/wmj/miniconda3/envs/sim/bin:'+env['PATH'],CPATH='/home/wmj/miniconda3/envs/splatad/include',TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),MAX_JOBS='4')
subprocess.run(['/home/wmj/miniconda3/envs/sim/bin/python',str(ROOT/'scripts/render_orbit.py'),'--ply',str(out/'canonical.ply'),'--output',str(out),'--length',str(case['metric_dimensions_lwh'][0]),'--distance','11'],env=env,check=True)
print('Completed generation, loader reload, 16 views and orbit; heading and quality review pending',flush=True)
