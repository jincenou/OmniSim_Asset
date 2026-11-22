"""Select the frozen R/F/H inputs and run the headless official backend."""
from pathlib import Path
import argparse,json,hashlib,traceback,math
from runtime import ROOT,inference

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',choices=['R','F','H'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--real-count',type=int,choices=[1,2]);a=p.parse_args()
    case=json.loads((ROOT/'inputs'/a.case/'segmented_manifest.json').read_text())
    accepted={'R':{'real'},'F':{'flux'},'H':{'real','flux'}}[a.variant]
    rows=[r for r in case['images'] if r['split']=='train' and r['source_type'] in accepted]
    tag=a.variant
    if a.real_count:
        if a.variant!='R':raise ValueError('Count ablation is real-only')
        reference=next(r for r in rows if r['id']==case['flux_reference_id'])
        selected=[reference]
        if a.real_count==2:
            def gap(row):
                delta=row['view_azimuth_rad']-reference['view_azimuth_rad']
                return abs(math.atan2(math.sin(delta),math.cos(delta)))
            selected.append(max([r for r in rows if r['id']!=reference['id']],key=gap))
        if len({r['id'] for r in selected})!=a.real_count:raise ValueError('Duplicate selected view')
        rows=selected;tag=f'R{a.real_count}'
    if not rows:raise ValueError('No training inputs')
    if a.variant in ('F','H') and sum(r['source_type']=='flux' for r in rows)!=2:raise ValueError('Expected the original FLUX pair')
    for row in rows:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Frozen RGB changed')
        if not Path(row['rgba_path']).is_file():raise FileNotFoundError(row['rgba_path'])
    heldout={r['sha256'] for r in case['images'] if r['split']=='heldout'}
    if any(r['sha256'] in heldout for r in rows):raise ValueError('Heldout leakage')
    output=ROOT/'results'/a.case/f'{tag}_seed{a.seed}'
    output.mkdir(parents=True,exist_ok=True)
    (output/'input_manifest.json').write_text(json.dumps(dict(case_id=a.case,variant=tag,seed=a.seed,images=rows),indent=2))
    try:
        print(json.dumps(inference([r['rgba_path'] for r in rows],output,a.seed),indent=2))
    except Exception:
        (output/'failure.txt').write_text(traceback.format_exc())
        raise

if __name__=='__main__':main()
