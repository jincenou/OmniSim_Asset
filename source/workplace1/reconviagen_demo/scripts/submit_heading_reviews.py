"""Persist explicit human/agent visual decisions and start only their export jobs."""
from pathlib import Path
import argparse,json,shlex,subprocess
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--sheet',required=True);p.add_argument('--flips',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--ambiguous-index',type=int);a=p.parse_args()
sheet=ROOT/'docs'/a.sheet;rows=json.loads(sheet.with_suffix('.json').read_text());flips=[int(x) for x in a.flips.split(',')]
if len(rows)!=len(flips) or any(x not in [0,1] for x in flips):raise ValueError('One explicit flip decision required per reviewed row')
path=ROOT/'inputs/random10_heading_decisions.json';reviews=json.loads(path.read_text());existing={(r['case_id'],r['variant'],r['seed']) for r in reviews}
lines=['#!/bin/bash','set -e',f'cd {ROOT}',f'export CUDA_VISIBLE_DEVICES={a.gpu} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1']
for i,(r,flip) in enumerate(zip(rows,flips)):
 key=(r['case_id'],r['variant'],r['seed'])
 if key in existing:raise ValueError('Already reviewed '+str(key))
 note='Visual front/rear light, grille and cabin check';confidence=1.
 if i==a.ambiguous_index:note='Ambiguous frontal appearance: rear-like texture on both ends. Heading assigned from hood/cabin geometry, not metric optimization; inspect qualitative result.';confidence=.5
 reviews.append(dict(case_id=r['case_id'],variant=r['variant'],seed=r['seed'],flip=bool(flip),evidence=str(sheet.with_suffix('.jpg')),confidence=confidence,note=note))
 common=['--case',r['case_id'],'--variant',r['variant'],'--seed',str(r['seed'])];tag=r['case_id']+'_'+r['variant']
 cmd=[str(ROOT/'env/bin/python'),'scripts/finalize_heading.py',*common,'--confidence',str(confidence),'--note',note]+(['--flip'] if flip else [])
 lines.append(shlex.join(cmd)+f' > logs/{tag}_heading.log 2>&1')
 lines.append(shlex.join(['/home/wmj/miniconda3/envs/sim/bin/python','scripts/render_metric_views.py',*common])+f' > logs/{tag}_metric_render.log 2>&1')
path.write_text(json.dumps(reviews,indent=2));job=ROOT/'scripts'/('job_'+sheet.stem+'.sh');job.write_text('\n'.join(lines)+'\n')
subprocess.run(['tmux','-L','reconviagen','new-session','-d','-s',sheet.stem,shlex.join(['bash',str(job)])],check=True)
print('Started explicit heading decisions',len(rows),sheet.stem)
